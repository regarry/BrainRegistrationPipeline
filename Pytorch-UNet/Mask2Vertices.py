import cv2
import os
import scipy.ndimage as ndi
import numpy as np
import matplotlib.pyplot as plt

minimum_coordinates = 100
dir_path = './brain_1/'
sub_dir_path = 'masks/'
image_format = '.png'

if not os.path.exists(f'{dir_path}/markers/'):
    os.makedirs(f'{dir_path}/markers/')
if not os.path.exists(f'{dir_path}/contours/'):
    os.makedirs(f'{dir_path}/contours/')
    
# Get a list of all image files in the directory with the specified format
image_files = []
for file in os.listdir(dir_path + sub_dir_path):
    if file.endswith(image_format):
        image_files.append(file)
image_file_paths = [os.path.join(dir_path + sub_dir_path, file) for file in image_files]
for file_path, file_name in zip(image_file_paths, image_files):
    mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # display the contours and coordinates in png files
    contour = np.zeros(mask.shape, dtype='uint8')
    marker = np.zeros(mask.shape, dtype='uint8')
    for i in range(len(contours)):
        coordinates = np.array(contours[i]).reshape((-1,2))
        if coordinates.shape[0] > minimum_coordinates:
            cv2.drawContours(contour, contours=contours, contourIdx=i, color=(255, 255, 255), thickness=1)
            for coordinate in coordinates:
                cv2.drawMarker(marker,coordinate, color=(255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=1, thickness=1)
        
    cv2.imwrite(f'{dir_path}/markers/{file_name[:len(file_name)-len(image_format)]}_markers.png', marker)
    cv2.imwrite(f'{dir_path}/contours/{file_name[:len(file_name)-len(image_format)]}_contours.png', contour)

    # make the combined coordinate array
    total_coordinates = np.full((2, 2), np.nan)
    for i in range(len(contours)):
        coordinates = np.array(contours[i]).reshape((-1,2))
        if coordinates.shape[0] > minimum_coordinates:
            print(coordinates.shape)
            total_coordinates = np.append(total_coordinates, coordinates, axis=0) 
    total_coordinates = total_coordinates[2:,:] # remove the first two rows of nan
    print(total_coordinates.shape)
