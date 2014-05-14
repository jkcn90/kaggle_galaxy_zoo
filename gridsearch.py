from galaxy_data import GalaxyData
from sklearn import (grid_search, ensemble)

import evaluate

data = GalaxyData(scale_features=False)
(X_train, y_train) = data.get_training_data()

clf = ensemble.RandomForestRegressor(n_estimators=100, max_features='log2')

parameters = {'max_depth': [5, 10, 15, 20, 25, 30]} 

gs = grid_search.GridSearchCV(clf, param_grid=parameters, scoring=evaluate.get_rmse_clf, n_jobs=-1,
        cv=5, verbose=5)
gs.fit(X_train, y_train)
