import os
import numpy as np
import SimpleCV as cv

def physical(path):
    """Extract features from a jpeg.

    Args:
        path: A string corresponding to the location of the jpeg.

    Returns:
        A list of numbers representing features.
    """
    img = cv.Image(path)

    # Find the largest blob in the image and crop around it
    blobs = img.findBlobs()
    largest_blob = blobs.filter(blobs.area() == max(blobs.area()))

    cropped_image = largest_blob.crop()[0]
    cropped_image = cropped_image.toHLS()

    # Feature vector: [aspect_ratio, hue, lightness, saturation]
    # Remember to update labels if features are updated here
    aspect_ratio = largest_blob[0].aspectRatio()
    (feature_hue, feature_lightness, feature_saturation) = cropped_image.meanColor()
    feature_vector = [aspect_ratio, feature_hue, feature_lightness, feature_saturation]

    feature_vector = _add_galaxy_id(path, feature_vector)
    return feature_vector

def raw(path):
    img = cv.Image(path)

    # Find the largest blob in the image and crop around it
    blobs = img.findBlobs()
    largest_blob = blobs.filter(blobs.area() == max(blobs.area()))[0]

    # Rotate blob
    angle = largest_blob.angle()
    w = largest_blob.minRectWidth()
    h = largest_blob.minRectHeight()

    if w < h:
        angle -= 90

    img = img.rotate(angle)

    # Get the bounding box of the image and calculate a centered square
    bounding_box_xywh = largest_blob.boundingBox()

    center = largest_blob.centroid()
    max_dim = max(bounding_box_xywh[2], bounding_box_xywh[3])
    xywh = center+(max_dim, max_dim)
    cropped_image = img.crop(xywh, centered=True)

    # Return the raw array scaled to a feasible size
    #cropped_image = cropped_image.toHLS()
    cropped_image = cropped_image.resize(3, 3)
    raw_array = cropped_image.getNumpy()
    raw_array = raw_array

    feature_vector = raw_array.flatten()

    feature_vector = _add_galaxy_id(path, feature_vector)
    return feature_vector

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
