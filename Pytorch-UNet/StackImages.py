import numpy as np
import os
from scipy.ndimage import rotate
from PIL import Image
import matplotlib.pyplot as plt
from ipywidgets import interact

def plot_slice(slice):
    plt.imshow(slice, cmap='gray')
    plt.show()

def explore_3dimage(volume):
    interact(lambda slice_index: plot_slice(volume[:, :, slice_index]), 
             slice_index=(0, volume.shape[2] - 1))
    
def stack_images_to_volume(dir_path, image_format):
    # Get a list of all image files in the directory with the specified format
    image_files = [os.path.join(dir_path, file) for file in os.listdir(dir_path) if file.endswith(image_format)]

    # Load 2D images
    images = [Image.open(image_file) for image_file in image_files]

    # Stack 2D images into a 3D volume
    volume = np.stack(images, axis=-1)

    return volume

def slice_volume(volume, plane, index):
    if plane == 'xy':
        slice = volume[:, :, index]
    elif plane == 'xz':
        slice = volume[:, index, :]
    elif plane == 'yz':
        slice = volume[index, :, :]
    else:
        raise ValueError("Invalid plane; choose from 'xy', 'xz', 'yz'")

    return slice

def oblique_slice(volume, angle, axis, slice_index):
    # Rotate the volume
    rotated_volume = rotate(volume, angle, axes=axis, reshape=True)

    # Slice the volume
    oblique_slice = rotated_volume[slice_index]

    return oblique_slice

from PIL import Image

def save_slice_as_png(slice, file_name):
    # Convert to PIL Image and save
    img = Image.fromarray(slice)
    img.save(file_name)

# Usage
volume = stack_images_to_volume('./brain_1_masks/', '.png')
print(volume.shape)
oblique = oblique_slice(volume, 45, (1, 2), 50)
print(oblique.shape)
save_slice_as_png(oblique, 'output_reshaped.png')

