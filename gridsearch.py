from galaxy_data import GalaxyData
from sklearn import (grid_search, ensemble)

import evaluate
import pickle

data = GalaxyData(scale_features=False)
(X_train, y_train) = data.get_training_data()

clf = ensemble.RandomForestRegressor(n_estimators=100, max_features='log2')

parameters = {'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10]}

gs = grid_search.GridSearchCV(clf, param_grid=parameters, scoring=evaluate.get_rmse_clf, n_jobs=-1,
        cv=5, verbose=5)
gs.fit(X_train, y_train)
print(gs.grid_scores_)
pickle.dump(gs, open( "min_samples_split_rf", "wb" ))
