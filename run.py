import os
import glob
import math
import argparse
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
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
     validation_features, validation_solutions) = data.split_training_and_validation_data(90)

    # Run Machine Learning Algorithm
    clf = RandomForestRegressor(1000, n_jobs=-1, verbose=5)
    #clf = ExtraTreesRegressor(n_jobs=-1, verbose=1)
    #clf = DecisionTreeRegressor()
    #clf = AdaBoostRegressor(base_estimator=clf, n_estimators=10, loss='exponential')

    #clf = GradientBoostingRegressor()
    #clf = svm.SVC()

    print('Training Model... ')
    clf.fit(training_features, training_solutions)
    print('Done Training')

    print('Predicting...')
    predicted_validation_solutions = clf.predict(validation_features)
    predicted_validation_solutions = pd.DataFrame(predicted_validation_solutions,
                                                  index=validation_features.index,
                                                  columns=validation_solutions.columns)
    print('Done Predicting')
    rmse = math.sqrt(mean_squared_error(validation_solutions.values,
                                        predicted_validation_solutions.values))
    print('RMSE: ' + str(100*rmse) + '%')
    return (validation_solutions, predicted_validation_solutions)

def run_test():
    # Load the data
    data = GalaxyData()

    (training_features, training_solutions) = data.get_training_data()
    test_features = data.get_test_data()

    # Evaluate
    clf = RandomForestRegressor(1, n_jobs=-1, verbose=5)
    print('Training Model...')
    clf.fit(training_features, training_solutions)
    print('Done Training')

    print('Predicting...')
    predicted_solutions = clf.predict(test_features)
    predicted_solutions = pd.DataFrame(predicted_solutions,
                                       index=test_features.index,
                                       columns=training_solutions.columns)
    print('Done Predicting')
    predicted_solutions.to_csv('output_data/a.csv')

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
        run_test()
