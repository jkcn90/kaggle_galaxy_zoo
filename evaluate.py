import math

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
