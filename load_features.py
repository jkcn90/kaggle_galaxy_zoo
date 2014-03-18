import os
import glob
import pandas as pd
import SimpleCV as cv

CURSOR_UP_ONE = '\x1b[1A'
ERASE_LINE = '\x1b[2K'

def extract_features_from_image(path):
    """Extract features from a jpeg.

    Args:
        path: A string corresponding to the location of the jpeg.

    Returns:
        A list of numbers representing features.
    """
    img = cv.Image(path)

    # Find the largest blob in the image and crop around it
    blobs = img.findBlobs()
    largest_blob = blobs.filter(blobs.area() == max(blobs.area()))

    cropped_image = largest_blob.crop()[0]
    cropped_image = cropped_image.toHLS()

    # Feature vector: [area, hue, lightness, saturation]
    feature_area = cropped_image.area()
    (feature_hue, feature_lightness, feature_saturation) = cropped_image.meanColor()
    feature_vector = [feature_area, feature_hue, feature_lightness, feature_saturation]
    return feature_vector

def extract_features_from_directory(input_directory):
    """Extract features from all jpegs in a given directory.

    Args:
        input_directory: A string corresponding to the directory where the jpeg's are located.

    Returns:
        A pandas DataFrame object containing the feature vectors.
    """
    glob_path = os.path.join(input_directory, '*.jpg')
    jpg_files = glob.glob(glob_path)
    
    number_of_images = len(jpg_files)
    feature_vector_list = []

    for (count, jpg_file) in enumerate(jpg_files):
        percent_done = (10000*count / number_of_images) / 100.0
        print('Processing Image: ' + str(count+1) + '/' + str(number_of_images) + 
              ' (' + str(percent_done) + '%)')

        # Extract a feature vector from each jpg
        galaxy_id = int(os.path.splitext(os.path.basename(jpg_file))[0])
        feature_vector = extract_features_from_image(jpg_file)
        feature_vector_list.append([galaxy_id] + feature_vector)
        print(CURSOR_UP_ONE + ERASE_LINE + CURSOR_UP_ONE)

    # Convert feature vector list to a data frame object and label it. Remember to update labels
    # here if features are changed.
    columns = ['GalaxyID', 'area', 'hue', 'lightness', 'saturation']
    feature_vectors = pd.DataFrame(data=feature_vector_list, columns=columns,)
    feature_vectors.set_index('GalaxyID', inplace=True)
    feature_vectors.index.name=None
    return feature_vectors

def main(input_directory, output_file):
    """Extract feature vectors from a directory of jpeg's and save a copy to file.

    If the feature vectors in question have already been serialized, load them in.
    
    Args:
        input_directory: A string corresponding to the directory where the jpeg's are located.
        output_file: A string corresponding to the path where the feature vectors will be saved.

    Returns:
        A pandas DataFrame object containing the feature vectors.
    """
    if os.path.exists(output_file):
        print('Loading: ' + output_file)
        feature_vectors = pd.read_pickle(output_file)
    else:
        print('Extracting Feature Vectors From Images:')
        feature_vectors = extract_features_from_directory(input_directory)

        # Serialize the feature vector so we can try different algorithms without running the
        # feature extraction process again
        print('WRITING: ' + output_file)
        feature_vectors.to_pickle(output_file)
        print('DONE WRITING')
    return feature_vectors

if __name__ == '__main__':
    feature_vectors = main('input_data/images_training_rev1', 'output_data')
    print(feature_vectors)
