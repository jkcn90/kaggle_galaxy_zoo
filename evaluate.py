import math

from sklearn.metrics import mean_squared_error

def get_rmse(actual, predicted):
    rmse = math.sqrt(mean_squared_error(actual, predicted))
    print('RMSE: ' + str(100*rmse) + '%')
    return rmse
