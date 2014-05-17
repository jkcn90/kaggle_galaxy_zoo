'''
Created on Apr 13, 2014

@author: darshanhegde

@note: Does rgb2grey, central cropping and sub-sampling of the original image
'''

import os
import random
from multiprocessing import Pool
from skimage import io
from skimage import color
from skimage.transform import resize


in_folder_path = "input_data/images_training_rev1/"
out_folder_path = "hog_data/images_training_rev1/"
        
            
def process(galaxy_image_name):
    global in_folder_path, out_folder_path
    galaxy_image = io.imread(os.path.join(in_folder_path, galaxy_image_name), as_grey=True)
    galaxy = color.rgb2grey(galaxy_image)
    galaxy = galaxy[108:316, 108:316]
    galaxy_sub = resize(galaxy, (104, 104))
    io.imsave(os.path.join(out_folder_path, galaxy_image_name), galaxy_sub)

def main():
    galaxy_image_names = os.listdir(in_folder_path)
    pool = Pool()
    random.shuffle(galaxy_image_names)
    #pool.map(process, galaxy_image_names)
    map(process, galaxy_image_names)
            
if __name__ == '__main__':
    main()
