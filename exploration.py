import pandas as pd
from galaxy_data import GalaxyData

def get_reduced_solutions(solutions=None, upper_threshold=1):
    """Reduces solutions to only deal with class1.1, class1.2, and class1.3. Only values with one of
    these classes having more than the upper_threshold will be returned, all others will be dropped.

    Attributes:
        solutions: A DataFrame object of solutions. If None is passed in we will load the csv file.
        upper_threshold: If there is no class with value greater than the upper_threshold they will
                         be dropped.

    Returns: DataFrame of the reduced solutions.
    """
    if solutions is None:
        solutions = pd.read_csv("input_data/training_solutions_rev1.csv", index_col="GalaxyID")

    solutions = solutions[["Class1.1", "Class1.2", "Class1.3"]]
    
    df_index = solutions[(solutions >= upper_threshold)]
    df_index = df_index.dropna(how='all')
    index = df_index.index
    
    solutions = solutions.ix[index]
    return solutions

def extract_features(extraction_method, index=None, percent_subset=100):
    """Runs the given extraction method on only those galaxys listed in index. Return a subset of
    those galaxies.

    Attrubutes:
        extraction_method: Extraction method to use. See feature_extraction
        index: Index of Galaxy for which to process data. If None, process all galaxies.
        percent_subset: Returns a subset of the data of this size (percent).

    Returns: A Tuple containing (X, y), with X being the features and y the labels.
    """
    data = GalaxyData(extraction_method, scale_features=False)
    if index is not None:
        data.set_restricted_universe(index)

    if percent_subset == 100:
        (X, y) = data.get_training_data()
    else:
        (X, y, _, _) = data.split_training_and_validation_data(100-percent_subset)

    y = get_reduced_solutions(y, -1)
    return (X, y)
