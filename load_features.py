from __future__ import print_function

import os
import glob
import numpy as np
import pandas as pd
import SimpleCV as cv
import multiprocessing

from PIL import Image
from sklearn import preprocessing

def extract_features_from_image_physical(path):
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
    return feature_vector

def extract_features_from_image_raw(path):
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
    cropped_image = cropped_image.toHLS()
    cropped_image = cropped_image.resize(20, 20)
    raw_array = cropped_image.getNumpy()
    raw_array = raw_array[:,:,1]

    feature_vector = raw_array.flatten()
    return feature_vector

def extract_features_from_image_loader(path):
    """Sets up parallel process to run feature extraction from a jpeg

    Args:
        path: A string corresponding to the location of the jpeg.

    Returns:
        A list of numbers representing features.
    """
    galaxy_id = int(os.path.splitext(os.path.basename(path))[0])

    # Extract features from images
    feature_vector = np.append(galaxy_id, extract_features_from_image_raw(path))
    return feature_vector

def extract_features_from_directory(input_directory):
    """Extract features from all jpegs in a given directory.

    Args:
        input_directory: A string corresponding to the directory where the jpeg's are located.

    Returns:
        A pandas DataFrame object containing the feature vectors.
    """
    glob_path = os.path.join(input_directory, '*.jpg')
    jpg_files = glob.glob(glob_path)
    number_of_images = len(jpg_files)

    # Setup parallel processing
    print('Extracting Features ...')
    pool_size = multiprocessing.cpu_count() * 2
    chunk_size = number_of_images // pool_size

    pool = multiprocessing.Pool(pool_size, maxtasksperchild=2)
    feature_vector_list = pool.map(extract_features_from_image_loader, jpg_files, chunk_size)

    pool.close()
    pool.join()
    print('Finished Extracting Features')

    # Convert feature vector list to a data frame object and label it. Remember to update labels
    # here if features are changed.
    #columns = ['GalaxyID', 'area', 'hue', 'lightness', 'saturation']
    #feature_vectors = pd.DataFrame(feature_vector_list, columns=columns)
    #feature_vectors.set_index('GalaxyID', inplace=True)
    feature_vectors = pd.DataFrame(feature_vector_list)
    feature_vectors.set_index(0, inplace=True)
    feature_vectors.index.name=None

    # Scale the parameters
    scaled_values = preprocessing.scale(feature_vectors.values.astype(float))
    feature_vectors = pd.DataFrame(scaled_values, feature_vectors.index, feature_vectors.columns)
    return feature_vectors

def main(input_directory, output_file):
    """Extract feature vectors from a directory of jpeg's and save a copy to file.

    If the feature vectors in question have already been serialized, load them in.
    
    Args:
        input_directory: A string corresponding to the directory where the jpeg's are located.
        output_file: A string corresponding to the path where the feature vectors will be saved.

    Returns:
        A pandas DataFrame object containing the feature vectors.
    """
    if os.path.exists(output_file):
        print('Loading: ' + output_file)
        feature_vectors = pd.read_pickle(output_file)
    else:
        print('Extracting Feature Vectors From Images:')
        feature_vectors = extract_features_from_directory(input_directory)

        # Serialize the feature vector so we can try different algorithms without running the
        # feature extraction process again
        print('WRITING: ' + output_file)
        feature_vectors.to_pickle(output_file)
        print('DONE WRITING')
    return feature_vectors

if __name__ == '__main__':
    feature_vectors_file = os.path.join('output', 'feature_vectors_training')
    feature_vectors = main('input_data/images_training_rev1', feature_vectors_file)
    print(feature_vectors)
