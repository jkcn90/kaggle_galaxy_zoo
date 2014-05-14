import math

from sklearn.metrics import mean_squared_error

def get_rmse(actual, predicted):
    """Calculates root mean square error.

    Args:
        actual: actual values.
        predicted: predicted values.
    """
    predicted.iloc[:,0:3].to_csv("HoGPredictions.csv")
    rmse = math.sqrt(mean_squared_error(actual.iloc[:,0:3], predicted.iloc[:,0:3]))
    print('RMSE: ' + str(100*rmse) + '%')
    return rmse
