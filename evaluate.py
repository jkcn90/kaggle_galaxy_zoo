import math
import numpy as np

from sklearn.metrics import mean_squared_error

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
        y_predicted = clf.predict(X_validate)
        rmse = math.sqrt(metrics.mean_squared_error(y_validate, y_predicted))
        rmse_list.append(rmse)
    rmse_list = np.array(rmse_list)
    return rmse_list
