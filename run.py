import os
import glob
import argparse
import pandas as pd
from sklearn import svm

import galaxy_data
from galaxy_data import GalaxyData

def run():
    """Entry Point
    """
    # Load the data
    data = GalaxyData()

    # Split into training and validation sets
    (training_features, training_solutions,
     validation_features, validation_solutions) = data.split_training_and_validation_data()

    # Run Machine Learning Algorithm
    print('Training Model...')
    clf = svm.LinearSVC()
    clf.fit(training_features, training_solutions[0].values)
    print('Done Training')

    print('Predicting...')
    predicted_validation_solutions = clf.predict(validation_features)
    predicted_validation_solutions = pd.DataFrame(predicted_validation_solutions,
                                                  index=validation_features.index)
    print('Done Predicting')
    difference = predicted_validation_solutions != validation_solutions
    error = (difference.sum() / len(difference)).values[0]
    print('Error rate: ' + str(error))

def clean():
    """Cleans up the workspace.
    """
    # Remove output directory
    galaxy_data.clean()

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
        run()
