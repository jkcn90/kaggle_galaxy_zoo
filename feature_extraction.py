import os
import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage as nd
try:
    import SimpleCV as cv
except:
    print('Warning: Running without SimpleCV')
from skimage import io
from skimage import color
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
from skimage.filter import threshold_otsu
from skimage.morphology import label
from skimage.measure import regionprops
from skimage import transform
from skimage.filter import gabor_kernel


def default(path):
    """Extract features from a jpeg.

    Args:
        path: A string corresponding to the location of the jpeg.

    Returns:
        A list of numbers representing features.
    """
    return raw(path)

def physical(path):
    img = cv.Image(path)

    # Find the largest blob in the image and crop around it
    blobs = img.findBlobs()
    largest_blob = blobs.filter(blobs.area() == max(blobs.area()))

    cropped_image = largest_blob.crop()[0]
    cropped_image = cropped_image.toHLS()

    # Feature vector: [aspect_ratio, hue, lightness, saturation]
    aspect_ratio = largest_blob[0].aspectRatio()
    (feature_hue, feature_lightness, feature_saturation) = cropped_image.meanColor()
    feature_vector = [aspect_ratio, feature_hue, feature_lightness, feature_saturation]

    feature_vector = _add_galaxy_id(path, feature_vector)
    return feature_vector

def raw_all(path):
    img = cv.Image(path)

    feature_vector = img.getGrayNumpy()

    feature_vector = _add_galaxy_id(path, feature_vector)
    return feature_vector

def raw(path, rotate_images=False, cropped_size=9):
    img = cv.Image(path)

    # Find the largest blob in the image and crop around it
    blobs = img.findBlobs()
    largest_blob = blobs.filter(blobs.area() == max(blobs.area()))[0]

    # Rotate blob
    if rotate_images:
        angle = largest_blob.angle()
        w = largest_blob.minRectWidth()
        h = largest_blob.minRectHeight()

        if w < h:
            angle -= 90

        img = img.rotate(angle)

    # Get the bounding box of the image and calculate a centered square
    bounding_box_xywh = largest_blob.boundingBox()

    center = largest_blob.centroid()
    max_dim = max(bounding_box_xywh[2], bounding_box_xywh[3])*1.3
    if max_dim <= 424:
        xywh = center+(max_dim, max_dim)
        cropped_image = img.crop(xywh, centered=True)
    else:
        cropped_image = img

    # Return the raw array scaled to a feasible size
    cropped_image = cropped_image.resize(cropped_size, cropped_size)
    raw_array = cropped_image.getNumpy()

    feature_vector = raw_array.flatten()

    feature_vector = _add_galaxy_id(path, feature_vector)
    return feature_vector

def hog_features(path):
    '''
    Takes the image path and uses skimage libraries to get HoG features.
    @author: Darshan Hegde
    '''
    print "Processing image: ", path.split("/")[-1]
    galaxy_image = io.imread(path, as_grey=True)
    galaxy_image = exposure.rescale_intensity(galaxy_image, out_range=(0,255))    # Improving contrast
    galaxy_image = rotateImage(galaxy_image)
    galaxy_image = denoise_tv_chambolle(galaxy_image, weight=0.15)
    fd = hog(galaxy_image, orientations=8, pixels_per_cell=(12, 12),
                    cells_per_block=(1, 1), visualise=False)

    print fd.shape
    feature_vector = _add_galaxy_id(path, fd)
    return feature_vector

def compute_gabor_feats(image, kernels):
    '''
    Compute gabor kernel features
    '''
    feats = np.zeros(4*len(kernels), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered_real = nd.convolve(image, np.real(kernel), mode='wrap')
        feats[4*k] = filtered_real.mean()
        feats[4*k+1] = filtered_real.var()
        filtered_img = nd.convolve(image, np.imag(kernel), mode='wrap')
        feats[4*k+2] = filtered_real.mean()
        feats[4*k+3] = filtered_real.var()
    return feats

            
def gabor_kernel_features(path):
    print "Processing image: ", path.split("/")[-1]
    galaxy_image = io.imread(path, as_grey=True)
    galaxy_image = exposure.rescale_intensity(galaxy_image, out_range=(0,255))    # Improving contrast
    galaxy_image = rotateImage(galaxy_image)
    kernels = []
    for theta in [0, 1, 2, 3]:
        theta = theta / 4. * np.pi
        for frequency in (0.05, 0.1, 0.2, 03, 0.4, 0.5, 0.6, 0.7):
            kernel = gabor_kernel(frequency, theta=theta)
            kernels.append(kernel)
    galaxy_image = (galaxy_image-galaxy_image.mean())/galaxy_image.std()
    feature_vector = compute_gabor_feats(galaxy_image, kernels)
    feature_vector = _add_galaxy_id(path, feature_vector)
    return feature_vector
    

def rotateImage(inImage):
    '''
    Takes the biggest connected component and finds the orientation and
    rotates so that the ellipse is vertical
    @author: Darshan Hegde
    '''
    # apply threshold
    thresh = threshold_otsu(inImage)
    thImage = inImage > thresh
    
    # label image regions
    label_image = label(thImage)
    
    max_region = None
    max_area = 0
    for region in regionprops(label_image):
    
        if max_area<region.area:
            max_area = region.area
            max_region = region
    
    rtImage = transform.rotate(inImage, 90-(180*max_region.orientation/np.pi))
    return rtImage
    

def _add_galaxy_id(path, feature_vector):
    """Adds galaxy id to a feature vector.

    Args:
        path: A string corresponding to the location of the jpeg.
        feature_vector: A list of numbers representing features.

    Returns:
        A list of numbers representing features.
    """
    galaxy_id = int(os.path.splitext(os.path.basename(path))[0])
    feature_vector = np.append(galaxy_id, feature_vector)
    return feature_vector
