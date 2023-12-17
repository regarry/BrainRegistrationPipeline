import numpy as np
import os
from scipy.ndimage import rotate
from scipy.io import savemat
from PIL import Image
import matplotlib.pyplot as plt
from ipywidgets import interact

def create_cube(volume_size, cube_size):
    # creates a solid cube with side length cube_size of 255s in a zero matrix of size array_size
    start_index = (volume_size - cube_size) // 2
    end_index = start_index + cube_size

    # Create a 3D array of zeros
    arr = np.zeros((volume_size, volume_size, volume_size),dtype=np.uint8)

    # Set a central subarray to 1
    arr[start_index:end_index, start_index:end_index, start_index:end_index] = 255

    return arr

def plot_slice(slice):
    plt.imshow(slice, cmap='gray')
    plt.show()

def explore_3dimage(volume):
    interact(lambda slice_index: plot_slice(volume[:, :, slice_index]), 
             slice_index=(0, volume.shape[2] - 1))
    
def stack_images_to_volume(dir_path, image_format):
    # Get a list of all image files in the directory with the specified format
    image_files = [os.path.join(dir_path, file) for file in os.listdir(dir_path) if file.endswith(image_format)]
    image_files = sorted(image_files)
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

def oblique_slice(volume, angle, axis, image_format, output_path):
    # Rotate the volume
    rotated_volume = rotate(volume, angle, axes=axis, reshape=False)
    #print(f'Rotated volume shape: {rotated_volume.shape}')
    # Slice the volume
    for i in range(rotated_volume.shape[2]):
        oblique_slice = rotated_volume[:,:,i]
        
        save_slice_as_png(oblique_slice, output_path + f'{i:03}_mask' + image_format)
    #print(f'Final oblique slice shape: {oblique_slice.shape}')
    #print(f'Number of slices generated: {i+1}')
    
def save_slice_as_png(slice, file_name):
    # Convert to PIL Image and save
    img = Image.fromarray(slice)
    img.save(file_name)
 
if __name__ == "__main__":    
    image_format = '.png'
    input_path = './brain_1/masks/'
    output_dir = './cube_1/'
    angle = 45
    dimensions = [1,2]
    output_path = output_dir + f'masks_{angle}_{dimensions[0]}{dimensions[1]}/'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # volume = stack_images_to_volume(input_path, image_format)
    # savemat('volume.mat', {'volume': volume})
    volume = create_cube(512,256)
    print(f'Input volume shape: {volume.shape}')
    oblique_slice(volume, angle, dimensions, image_format, output_path)
    output = stack_images_to_volume(output_path, image_format)
    savemat(output_path + f'cube_{angle}_{dimensions[0]}{dimensions[1]}.mat', {'output': output, 'input': volume})