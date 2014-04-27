import pandas as pd

from sklearn.ensemble import RandomForestRegressor

def default_model(features, solutions, verbose=0):
    clf = RandomForestRegressor(10, max_features='log2', n_jobs=-1, verbose=verbose)
    columns = solutions.columns

    print('Training Model... ')
    clf.fit(features, solutions)
    print('Done Training')
    return (clf, columns)

def predict(clf, features, columns):
    """Get the predicted solutions and configure

    Args:
        clf: The classifier to use to predict the solutions.
        features: Features to predict solutions with.
        columns: Columns to label the solutions.

    Returns: Predicted solutions as a Pandas DataFrame.
    """
    print('Predicting...')
    predicted_solutions = clf.predict(features)
    predicted_solutions = pd.DataFrame(predicted_solutions, index=features.index,
                                       columns=columns)
    print('Done Predicting')
    return predicted_solutions
