import math
import numpy as np
from scipy.stats import entropy

from sklearn.metrics import (mean_squared_error, normalized_mutual_info_score)

def get_rmse(actual, predicted):
    """Calculates root mean square error.

    Args:
        actual: actual values.
        predicted: predicted values.
    """
    # Restrict actual to the columns of predicted
    actual = actual[predicted.columns[:]]
   
    rmse = math.sqrt(mean_squared_error(actual, predicted))
    print('RMSE: ' + str(100*rmse) + '%')
    return rmse

def get_rmse_clf(estimator, X, y):
    y_predicted = estimator.predict(X)
    rmse = math.sqrt(mean_squared_error(y, y_predicted))
    return rmse

def get_errors_clf(estimator, X, y):
    y_predicted = estimator.predict(X)
    rmse = math.sqrt(mean_squared_error(y, y_predicted))
    kl_divergence = np.average(entropy(y.T, y_predicted.T))
    return (rmse, kl_divergence)

def cross_validate(clf, X, y, cv=5):
    if cv < 2:
        raise Exception("cv must be greater than 2")
    index = X.index
    num_images = index.shape[0]
    slice_size = num_images / cv

    rmse_list = []
    
    for i in range(0, cv):
        # Setup indicies
        validate_index = index[i*slice_size:(i+1)*slice_size]
        train_index = index.drop(validate_index)

        X_validate = X.ix[validate_index]
        X_train = X.ix[train_index]
        
        y_validate = y.ix[validate_index]
        y_train = y.ix[train_index]
        
        # Get rmse
        clf.fit(X_train, y_train)
        rmse = get_rmse_clf(clf, X_validate, y_validate)
        rmse_list.append(rmse)
    rmse_list = np.array(rmse_list)
    return rmse_list
