import os
import shutil
import random
import pandas as pd

import load_features
import feature_extraction

from sklearn import cross_validation

INPUT_DIRECTORY = '../../data/proc_images/'
OUTPUT_DIRECTORY = '../../data/proc_images/'

class GalaxyData:
    """Loads and processes the data required for operating on the GalaxyZoo data.

    This class will manage the output directory where pickled data files for the GalaxyZoo data will
    live. It will also control the extraction of features from the raw image files.

    Attributes:
        training_images_directory: The location of the training images directory
        test_images_directory: The location of the test images directory
        solutions_csv: The location of the training solutions csv file
    """

    def __init__(self, feature_extraction_func=feature_extraction.default, scale_features=True,
                 lle=False):
        """Initializes the GalaxyData class with file directories and file locations. Ensures the
           creation of the output directory.
        """
        self.feature_extraction_func = feature_extraction_func
        self.output_directory = os.path.join(OUTPUT_DIRECTORY, self.feature_extraction_func.__name__)

        self.training_images_directory = os.path.join(INPUT_DIRECTORY, 'images_training_rev1')
        self.test_images_directory = os.path.join(INPUT_DIRECTORY, 'images_test_rev1')

        self.solutions_csv = os.path.join(INPUT_DIRECTORY, 'training_solutions_rev1.csv')

        self.restricted_universe = None
        self.scale_features = scale_features
        self.lle = lle

        self.folder_setup()

    def set_restricted_universe(self, restricted_universe):
        """Sets restricted universe of GalaxyID to load

        Args:
            restricted_universe: Restricted universe of GalaxyID.
        """
        self.restricted_universe = restricted_universe

    def get_training_data(self, competition=False):
        """Gets the feature vectors and solutions for the training data.

        Args:
            competition: Trigger to determine competition or regular function

        Returns: A tuple containing (feature_vectors, solutions)
        """
        feature_vectors_file = os.path.join(self.output_directory, 'feature_vectors_training')
        (feature_vectors, solutions) = self._get_data(self.training_images_directory,
                                                      feature_vectors_file, competition)

        if not competition:
            (index, _) = self.get_training_test_index()
            feature_vectors = feature_vectors.ix[index]
            solutions = solutions.ix[index]
            
        return (feature_vectors, solutions)

    def get_test_data(self, competition=False):
        """Gets the feature vectors and solutions for the test data.

        Args:
            competition: Trigger to determine competition or regular function

        Returns: A tuple containing (feature_vectors, solutions)
        """
        if competition:
            feature_vectors_file = os.path.join(self.output_directory, 'feature_vectors_test')
            (feature_vectors, _) = self._get_data(self.test_images_directory, feature_vectors_file,
                                                  competition)
            solutions = None
        else:
            feature_vectors_file = os.path.join(self.output_directory, 'feature_vectors_training')
            (feature_vectors, solutions) = self._get_data(self.training_images_directory,
                                                          feature_vectors_file)
            (_, index) = self.get_training_test_index()
            feature_vectors = feature_vectors.ix[index]
            solutions = solutions.ix[index]
        return (feature_vectors, solutions)

    def get_training_test_index(self, test_size=0.5, random_state=0):
        index = pd.read_csv(self.solutions_csv, index_col='GalaxyID').index
        (training_index,
         test_index) = cross_validation.train_test_split(index, test_size=test_size,
                                                         random_state=random_state)
        return (training_index, test_index)

    def _get_data(self, images_directory, feature_vectors_file, competition=False):
        """Gets the feature vectors and solutions for the specified data.

        Loads the feature vectors directly from the images, or if they have already been loaded,
        read them in. Process the solutions.

        Returns: A tuple containing (feature_vectors, solutions)
        """
        feature_vectors = load_features.main(images_directory, feature_vectors_file,
                                             self.feature_extraction_func,
                                             self.scale_features,
                                             self.restricted_universe,
                                             self.lle)

        solutions = pd.read_csv(self.solutions_csv, index_col='GalaxyID')

        # Align the solutions to the GalaxyID of the feature_vectors
        solutions = solutions.ix[feature_vectors.index]
        if not competition:
            solutions = solutions[['Class1.1', 'Class1.2', 'Class1.3']]

        return (feature_vectors, solutions)

    def split_training_and_validation_data(self, percent_validation=25, seed=None, competition=False):
        """Splits the training data into training and validation data

        Args:
            percent_validation: The percentage of data to put in the validation set.
            seed: Random seed to use.

        Returns: A tuple containing the training_features, training_solutions, validation_features,
                 validation_solutions.
        """
        percent_validation /= 100.0
        (feature_vectors, solutions) = self.get_training_data(competition=competition)


        # Randomly split the training and validation data
        random.seed(seed)
        number_of_validation_rows = int(len(feature_vectors.index)*percent_validation)
        validation_rows = random.sample(feature_vectors.index, number_of_validation_rows)

        
        training_features = feature_vectors.drop(validation_rows)
        training_solutions = solutions.drop(validation_rows)

        validation_features = feature_vectors.ix[validation_rows]
        validation_solutions = solutions.ix[validation_rows]
        return (training_features, training_solutions, validation_features, validation_solutions)

    def save_solution(slef, solutions):
        """Save the solutions to csv.

        Args:
            solutions: Pandas representation of the solutions.
        """
        solutions_csv = os.path.join(OUTPUT_DIRECTORY, 'solutions.csv')
        solutions.to_csv(solutions_csv)

    def folder_setup(self):
        """Checks for the input directory and ensures that the output directory exists.
        """
        if not os.path.exists(INPUT_DIRECTORY):
            raise RuntimeError('Cannot find input directory: ' + INPUT_DIRECTORY)
        
        if not os.path.exists(self.output_directory):
            print('Creating output directory: ' + self.output_directory)
            os.makedirs(self.output_directory)


def clean():
    """Removes the output directory
    """
    if os.path.exists(OUTPUT_DIRECTORY):
        print('Removing output directory: ' + OUTPUT_DIRECTORY)
        shutil.rmtree(OUTPUT_DIRECTORY)
