import os
import glob
import argparse

import models
import evaluate
import galaxy_data
import feature_extraction

from galaxy_data import GalaxyData

def run(model, verbose=0):
    """Entry Point to run models

    Args:
        model: model function to run.
    """
    # Load the data and split into training and validation sets
    data = GalaxyData(feature_extraction.physical)

    (training_features, training_solutions,
     validation_features, validation_solutions) = data.split_training_and_validation_data(90)

    # Train and Predict Model
    (clf, columns) = model(training_features, training_solutions, verbose)
    predicted_validation_solutions = models.predict(clf, validation_features, columns)

    # Evaluate Predictions
    evaluate.get_rmse(validation_solutions, predicted_validation_solutions)

def competition_run():
    pass

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
        print('\n\t'.join(models_list))
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
    parser.add_argument('--competition', help='runs competition mode', action='store_true')
    parser.add_argument('--verbose', help='triggers extra information', action='store_true')
    parser.add_argument('--clean', help='cleans workspace', action='store_true')
    parser.add_argument('--list', help='lists available models', action='store_true')
    parser.add_argument('--model', help='runs the selected model')
    args = parser.parse_args()

    if args.list:
        list_models()
    elif args.clean:
        clean()
    elif args.competition:
        competition_run()
    elif args.verbose:
        run(resolve_model_name(args.model), 5)
    else:
        run(resolve_model_name(args.model))
