import pandas as pd


from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, MultiTaskLasso
from sklearn import svm
from numpy import arange
import matplotlib.pyplot as plt
import numpy as np

def default_model(features, solutions, verbose=0):
    return mulri_task_lasso(features, solutions, verbose)

def test_model(features, solutions, verbose=0):
    columns = solutions.columns[:1]
    solutions = solutions[columns[0]]

    clf = svm.SVR(max_iter=100, verbose=verbose)

    print('Training Model... ')
    clf.fit(features, solutions)
    print('Done Training')
    return (clf, columns)

def decision_tree_regressor(features, solutions, verbose=0):
    columns = solutions.columns

    clf = DecisionTreeRegressor(max_depth=8)

    print('Training Model... ')
    clf.fit(features, solutions)
    print('Done Training')
    
    features_importance = clf.feature_importances_
    features_importance = np.reshape(features_importance, (169, 8))
    features_importance = np.sum(features_importance, axis=1)
    features_importance = np.reshape(features_importance, (13, 13))
    plt.pcolor(features_importance)
    plt.show()
    
    return (clf, columns)
    

def ada_boost_model(features, solutions, verbose=0):
    columns = solutions.columns

    clf = DecisionTreeRegressor(max_depth=10)
    clf = AdaBoostRegressor(clf, n_estimators=20)

    print('Training Model... ')
    clf.fit(features, solutions)
    print('Done Training')
    return (clf, columns)

def gradient_boost_model(features, solutions, verbose=0):
    columns = solutions.columns[:1]
    solutions = solutions[columns[0]]

    clf = GradientBoostingRegressor(loss='ls', max_features='auto', verbose=verbose)

    print('Training Model... ')
    clf.fit(features, solutions)
    print('Done Training')
    return (clf, columns)

def random_forest_model(features, solutions, verbose=0):
    columns = solutions.columns

    clf = RandomForestRegressor(100, max_features='sqrt', n_jobs=-1, verbose=verbose)

    print('Training Model... ')
    clf.fit(features, solutions)
    print('Done Training')
    
#     features_importance = clf.feature_importances_
#     print(np.min(features_importance))
#     print(np.max(features_importance))
#     avg_gini_idx = np.average(features_importance)
#     print(np.average(features_importance))
#     fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 8))
#     features_importance = np.reshape(features_importance, (169, 8))
#     for idx, feat in enumerate(features_importance.T):
#         features_importance = np.reshape(feat, (13, 13))
#         print(idx)
#         axes[idx/4, idx%4].pcolor(features_importance.T, vmin=0, vmax=avg_gini_idx)
#          
#     plt.show()
    return (clf, columns)

def knn_regressor(features, solutions, verbose=0):
    columns = solutions.columns

    clf = KNeighborsRegressor(n_neighbors=5, weights='distance')

    print('Training Model... ')
    clf.fit(features, solutions)
    print('Done Training')
    return (clf, columns)

def mulri_task_lasso(features, solutions, verbose=0):
    columns = solutions.columns

    clf = MultiTaskLasso()

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
