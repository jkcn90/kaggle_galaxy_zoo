import pandas as pd

from sklearn.ensemble import RandomForestRegressor

def default_model(features, solutions):
    #clf = RandomForestRegressor(10, max_features='log2', n_jobs=-1, verbose=5)
    clf = RandomForestRegressor(10, max_features='log2', n_jobs=-1)
    columns = solutions.columns

    print('Training Model... ')
    clf.fit(features, solutions)
    print('Done Training')
    return (clf, columns)

def predict(clf, features, columns):
    print('Predicting...')
    predicted_solutions = clf.predict(features)
    predicted_solutions = pd.DataFrame(predicted_solutions, index=features.index,
                                       columns=columns)
    print('Done Predicting')
    return predicted_solutions
