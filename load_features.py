from __future__ import print_function

import os
import glob
import random
import pandas as pd
import multiprocessing

from sklearn import preprocessing

def extract_features_from_directory(input_directory, feature_extraction_func):
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

    pool_size = multiprocessing.cpu_count()*2
    chunk_size = number_of_images // pool_size
 
    pool = multiprocessing.Pool(pool_size, maxtasksperchild=2)
    feature_vectors = pool.map(feature_extraction_func, jpg_files, chunk_size)
 
    pool.close()
    pool.join()
    print('Finished Extracting Features')

    # Set GalaxyID as label
    feature_vectors = pd.DataFrame(feature_vectors)
    feature_vectors.set_index(0, inplace=True)
    feature_vectors.index.name='GalaxyID'

    # Scale the parameters
    scaled_values = preprocessing.scale(feature_vectors.values.astype(float))
    feature_vectors = pd.DataFrame(scaled_values, feature_vectors.index, feature_vectors.columns)
    return feature_vectors

def main(input_directory, output_file, feature_extraction_func):
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
        feature_vectors = extract_features_from_directory(input_directory, feature_extraction_func)

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
