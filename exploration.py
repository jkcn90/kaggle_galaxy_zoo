import pandas as pd
from galaxy_data import GalaxyData

def get_reduced_solutions(solutions=None, upper_threshold=1, lower_threshold=0, classification=False):
    """Reduces solutions to only deal with class1.1, class1.2, and class1.3. Only values with one of
    these classes having more than the upper_threshold will be returned, all others will be dropped.
    If classification trigger is on, class1.1, class1.2, and class1.3 will be mapped to 0, 1, 2.

    Attributes:
        solutions: A DataFrame object of solutions. If None is passed in we will load the csv file.
        upper_threshold: If there is no class with value greater than the upper_threshold they will
                         be dropped.
        lower_threshold: If there is no class with value less than the lower_threshold they will
                         be dropped.
        classification: Trigger to turn on classification for class1.1, class 1.2, and class 1.3.

    Returns: DataFrame of the reduced solutions.
    """
    if solutions is None:
        solutions = pd.read_csv("input_data/training_solutions_rev1.csv", index_col="GalaxyID")
        print_trigger = True
    else:
        print_trigger = False

    solutions = solutions[["Class1.1", "Class1.2", "Class1.3"]]
    
    index = solutions.applymap(lambda x: x <= upper_threshold and x >= lower_threshold)
    index_df = solutions[index]
    index_df = index_df.dropna(how='all')

    solutions = solutions.ix[index_df.index]
    if classification:
        solutions = solutions.apply(lambda x: 0 if x[0] >= upper_threshold
                                                else 1 if x[1] >= upper_threshold
                                                else 2, axis=1)
    if print_trigger:
        print('Processing: ' + str(solutions.shape[0]) + ' images...')
    return solutions

def extract_features(extraction_method, index=None, percent_subset=100, classification=False):
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

    y = get_reduced_solutions(y, classification=classification)
    return (X, y)
