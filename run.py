import os
import glob
import shutil
import argparse

import load_features

INPUT_DIRECTORY = 'input_data'
OUTPUT_DIRECTORY = 'output_data'

def folder_setup():
    """Checks for the input directory and ensures that the output directory exists.
    """
    if not os.path.exists(INPUT_DIRECTORY):
        print('Cannot find input directory: ' + INPUT_DIRECTORY)
        exit()
    
    if not os.path.exists(OUTPUT_DIRECTORY):
        print('Creating output directory: ' + OUTPUT_DIRECTORY )
        os.makedirs(OUTPUT_DIRECTORY)

def run():
    """Entry Point
    """
    image_directory_training = os.path.join(INPUT_DIRECTORY, 'images_training_rev1')
    feature_vector_path_training = os.path.join(OUTPUT_DIRECTORY, 'feature_vector_list')

    # Load the features vectors into the output directory
    feature_vector_training = load_features.main(image_directory_training,
                                                 feature_vector_path_training)

    # Run Machine Learning Algorithm

def clean():
    """Cleans up the workspace.
    """
    # Remove output directory
    if os.path.exists(OUTPUT_DIRECTORY):
        print('Removing output directory: ' + OUTPUT_DIRECTORY)
        print('Test Clean Function For Removing Output Directory')
        #shutil.rmtree(OUTPUT_DIRECTORY)

    # Remove files ending in pyc
    for pyc in glob.glob('*.pyc'):
        os.remove(pyc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean', help='cleans workspace',
                        action='store_true')
    args = parser.parse_args()

    if args.clean:
        clean()
    else:
        folder_setup()
        run()
