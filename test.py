import random
import numpy as np
import pandas as pd
import SimpleCV as cv

from sklearn import svm
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier

import evaluate
import feature_extraction
from galaxy_data import GalaxyData

solutions_raw = pd.read_csv("./input_data/training_solutions_rev1.csv", index_col="GalaxyID")
solutions = solutions_raw[["Class1.1", "Class1.2"]]

upper_threshold = 1

solutions = solutions[(solutions >= upper_threshold)]

solutions = solutions.dropna(how='all')
#solutions.apply(lambda x: x[0] if not isnan(x[0]) else x[1], axis=1).to_frame()
solutions = solutions.applymap(lambda x: 0 if np.isnan(x) else x)

data = GalaxyData(feature_extraction.raw)
data.set_restricted_universe(solutions.index)
(feature_vectors, _) = data.get_training_data()

