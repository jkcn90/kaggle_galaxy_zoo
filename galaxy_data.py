import os
import shutil
import random
import pandas as pd

import load_features

INPUT_DIRECTORY = 'input_data'
OUTPUT_DIRECTORY = 'output_data'

class GalaxyData:
    """Loads and processes the data required for operating on the GalaxyZoo data.

    This class will manage the output directory where pickled data files for the GalaxyZoo data will
    live. It will also control the extraction of features from the raw image files.

    Attributes:
        training_images_directory: The location of the training images directory
        test_images_directory: The location of the test images directory
        solutions_csv: The location of the training solutions csv file
    """

    def __init__(self):
        """Initializes the GalaxyData class with file directories and file locations. Ensures the
           creation of the output directory.
        """
        self.training_images_directory = os.path.join(INPUT_DIRECTORY, 'images_training_rev1')
        self.test_images_directory = os.path.join(INPUT_DIRECTORY, 'images_test_rev1')

        self.solutions_csv = os.path.join(INPUT_DIRECTORY, 'training_solutions_rev1.csv')

        self.folder_setup()

    def get_training_data(self):
        """Gets the feature vectos and solutions for the training data.

        Loads the feature vectors directly from the images, or if they have already been loaded,
        read them in. Process the solutions.

        Returns: A tuple containing (feature_vectors, solutions)
        """
        feature_vectors_file = os.path.join(OUTPUT_DIRECTORY, 'feature_vectors_training')
        feature_vectors = load_features.main(self.training_images_directory, feature_vectors_file)

        solutions = pd.read_csv(self.solutions_csv, index_col='GalaxyID')
        solutions.index.name=None
        solutions = solutions.ix[:,:'Class1.3']
        solutions = pd.DataFrame(solutions.idxmax(axis=1))
        return (feature_vectors, solutions)

    def split_training_and_validation_data(self, percent_validation=25, seed=None):
        """Splits the training data into training and validation data

        Args:
            percent_validation: The percentage of data to put in the validation set.
            seed: Random seed to use.

        Returns: A tuple containing the training_features, training_solutions, validation_features,
                 validation_solutions.
        """
        percent_validation /= 100.0
        (feature_vectors, solutions) = self.get_training_data()

        # Align the solutions to the GalaxyID of the feature_vectors
        solutions = solutions.ix[feature_vectors.index]

        # Randomly split the training and validation data
        random.seed(seed)
        number_of_validation_rows = int(len(feature_vectors.index)*percent_validation)
        validation_rows = random.sample(feature_vectors.index, number_of_validation_rows)

        
        training_features = feature_vectors.drop(validation_rows)
        training_solutions = solutions.drop(validation_rows)

        validation_features = feature_vectors.ix[validation_rows]
        validation_solutions = solutions.ix[validation_rows]
        return (training_features, training_solutions, validation_features, validation_solutions)

    def folder_setup(self):
        """Checks for the input directory and ensures that the output directory exists.
        """
        if not os.path.exists(INPUT_DIRECTORY):
            print('Cannot find input directory: ' + INPUT_DIRECTORY)
            exit()
        
        if not os.path.exists(OUTPUT_DIRECTORY):
            print('Creating output directory: ' + OUTPUT_DIRECTORY )
            os.makedirs(OUTPUT_DIRECTORY)


def clean():
    """Removes the output directory
    """
    if os.path.exists(OUTPUT_DIRECTORY):
        print('Removing output directory: ' + OUTPUT_DIRECTORY)
        shutil.rmtree(OUTPUT_DIRECTORY)
