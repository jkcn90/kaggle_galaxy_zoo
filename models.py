import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn import svm

def default_model(features, solutions, verbose=0):
    return test_model(features, solutions, verbose)

def test_model(features, solutions, verbose=0):
    clf = RandomForestRegressor(20, max_depth=25, max_features='log2', n_jobs=-1, verbose=verbose)
    columns = solutions.columns

    print('Training Model... ')
    clf.fit(features, solutions)
    print('Done Training')
    return (clf, columns)

def ideas():
    # Run Machine Learning Algorithm
    #clf = DecisionTreeRegressor()
    #clf = AdaBoostRegressor(base_estimator=clf, n_estimators=10, loss='exponential')
    #clf = GradientBoostingRegressor()
    #clf = svm.SVC()
    #clf = ExtraTreesRegressor(10, n_jobs=-1, verbose=5)
    pass

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
    predicted_solutions.to_csv("./results/predictions_hog.csv")
    print('Done Predicting')
    return predicted_solutions
