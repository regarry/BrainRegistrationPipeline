## this script will generate a variable sized cube of 
# value 255 within a empty 512x512x512 volume and slice 
# it using SliceVolume.py export those new slices to a 
# new folder and then use Mask2Vertices.py to convert 
# each slice to an array for model

import numpy as np
import os
from scipy.ndimage import rotate
from scipy.io import savemat
from PIL import Image
import SliceVolume as sv
from tqdm import tqdm
import time
import Mask2Vertices as m2v

volume_size = 512
cube_sizes = list(range(volume_size//2, volume_size, (volume_size-volume_size//2)//4))
image_format = '.png'
#input_path = './brain_1/masks/'
output_dir =  './cube_2/'
angles = list(range(10, 90, 20))
dimensions = [[0,1],[0,2],[1,2]]

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
#print(cube_sizes)
#print(angles)
#print(dimensions)

for cube_size in tqdm(cube_sizes):
    for angle in angles:
        for dimension in dimensions:
            output_path = output_dir + f'masks_{cube_size*100//volume_size:02}_{angle}_{dimension[0]}{dimension[1]}/'
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            volume = sv.create_cube(volume_size, cube_size)
            sv.oblique_slice(volume, angle, dimension, image_format, output_path)
            output = sv.stack_images_to_volume(output_path, image_format)
            savemat(output_path + f'cube_{angle}_{dimension[0]}{dimension[1]}.mat', {'output': output, 'input': volume})
            m2v.mask2vertices(output_path, image_format, 2)