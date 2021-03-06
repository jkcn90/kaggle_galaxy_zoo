import os
import glob
import argparse
import math

import models
import evaluate
import galaxy_data
import feature_extraction

from galaxy_data import GalaxyData
from sklearn import cross_validation
from sklearn import grid_search
from sklearn.metrics import mean_squared_error

def run(model, verbose=0):
    """Entry Point to run models

    Args:
        model: model function to run.
    """
    # Load the data and split into training and validation sets
    data = GalaxyData(feature_extraction.hog_features, scale_features=False)

    (training_features, training_solutions,
     validation_features, validation_solutions) = data.split_training_and_validation_data(50)

    # Train and Predict Model
    (clf, columns) = model(training_features, training_solutions, verbose)
    predicted_validation_solutions = models.predict(clf, validation_features, columns)

    # Evaluate Predictions
    valid_rmse = evaluate.get_errors_clf(clf, validation_features, validation_solutions)
    train_rmse = evaluate.get_errors_clf(clf, training_features, training_solutions)
    print " Validation RMSE: ", valid_rmse
    print " Training RMSE: ", train_rmse

def run_training_test(model, verbose=0):
    """Entry Point to run models

    Args:
        model: model function to run.
    """
    # Load the data and split into training and validation sets
    data = GalaxyData(feature_extraction.raw_9, scale_features=False)

    (test_features, test_solutions) = data.get_test_data()
    (training_features, training_solutions) = data.get_training_data()

    # Train and Predict Model
    (clf, columns) = model(training_features, training_solutions, verbose)
    predicted_solutions = models.predict(clf, test_features, columns)

    # Evaluate Predictions
    score = evaluate.get_rmse(test_solutions, predicted_solutions)
    print(score)

def competition_run():
    data = GalaxyData()

    (training_features, training_solutions) = data.get_training_data()
    (test_features, _) = data.get_test_data()

    # Predict
    (clf, columns) = models.default_model(training_features, training_solutions, 5)
    predicted_solutions = models.predict(clf, test_features, columns)

    data.save_solution(predicted_solutions)
    
def rmse_scorer(estimator, X, y):
    y_pred = estimator.predict(X)
    
    mse = mean_squared_error(y, y_pred)
    rmse = math.sqrt(mse)
    return rmse
    
def cross_validationcv(model, verbose=0):
    data = GalaxyData(feature_extraction.raw_1, scale_features=False)

    (features, solutions) = data.get_training_data()

    # Train and Predict Model
    (clf, _) = model(features, solutions, verbose)
    scores = cross_validation.cross_val_score(clf, features, solutions, cv=5, scoring=rmse_scorer, n_jobs=-1)
    print(scores)
    print("Cross validation error: ", sum(scores)/len(scores))
    
    
    
def grid_search_cv(model, verbose=0):
    data = GalaxyData(feature_extraction.hog_features, scale_features=False)

    (features, solutions) = data.get_training_data()

    # Train and Predict Model
    (clf, _) = model(features, solutions, verbose)
    parameters = {'min_sample_split': [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 1e-3, 1e-2, 1e-1]} 

    gs = grid_search.GridSearchCV(clf, param_grid=parameters, scoring=rmse_scorer, n_jobs=-1,
            cv=5, verbose=5)
    gs.fit(features, solutions)
    print(gs.grid_scores_)
    

def resolve_model_name(name):
    """Gets the model function corresponding to the name.

    Args:
        name: name of the model function to call.

    Returns: The model function corresponding to the name.
    """
    if name is None:
        name = 'default_model'
    else:
        models_list = list_models(False)
        if name not in models_list:
            raise RuntimeError('\n\t'.join(["Invalid Model: '" + name + 
                                            "'. Select from the following models:"] + models_list))
    model = getattr(models, (name))
    return model

def list_models(print_names=True):
    """Displays the available models in the model file.

    Args:
        print_names: Trigger to configure if models_list should be displayed.

    Returns: A list containing the available function models.
    """
    models_list = [function_names for function_names in dir(models) if 'model' in function_names]
    if print_names:
        print('\n\t'.join(['The following models are available:'] + models_list))
    return models_list

def clean():
    """Cleans up the workspace.
    """
    # Remove output directory
    galaxy_data.clean()

    # Remove files ending in pyc
    for pyc in glob.glob('*.pyc'):
        os.remove(pyc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs models on the GalaxyZoo data. The default' +
                                                 'model is used if no model is selected')
    parser.add_argument('-c', '--competition', help='runs competition mode', action='store_true')
    parser.add_argument('-cv', '--crossvalidate', help='runs cross validation', action='store_true')
    parser.add_argument('-gs', '--gridsearch', help='runs grid search over height', action='store_true')
    parser.add_argument('-v', '--verbose', help='triggers extra information', action='store_true')
    parser.add_argument('-x', '--clean', help='cleans workspace', action='store_true')
    parser.add_argument('-l', '--list', help='lists available models', action='store_true')
    parser.add_argument('-m', '--model', help='runs the selected model')
    parser.add_argument('-tt', '--train_test', help='evaluate test error', action='store_true')
    args = parser.parse_args()

    if args.verbose:
        verbose = 5
    else:
        verbose = 0

    if args.list:
        list_models()
    elif args.clean:
        clean()
    elif args.competition:
        competition_run()
    elif args.crossvalidate:
        cross_validationcv(resolve_model_name(args.model), verbose)
    elif args.train_test:
        run_training_test(resolve_model_name(args.model), verbose)
    elif args.gridsearch:
        grid_search_cv(resolve_model_name(args.model), verbose)
    else:
        run(resolve_model_name(args.model), verbose)
