import numpy as np

from evaluate import cross_validate
from galaxy_data import GalaxyData

from sklearn import (ensemble, cross_validation)

data = GalaxyData(scale_features=False)
(X_train, y_train) = data.get_training_data()
(X_test, y_test) = data.get_test_data()

clf = ensemble.RandomForestRegressor(n_estimators=10, n_jobs=-1, verbose=5)

scores = cross_validate(clf, X_train, y_train, 5)
mean_score = sum(scores) / float(scores.shape[0])
print(scores)
print(mean_score)
