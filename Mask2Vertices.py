import cv2
import os
import numpy as np
import shutil
from scipy.io import savemat, loadmat  
import argparse
import json


def mask2vertices(mask_path, image_format:  str = ".png", minimum_coordinates: int = 100):
    if not os.path.exists(f'{mask_path}/markers/'):
        os.makedirs(f'{mask_path}/markers/')
    if not os.path.exists(f'{mask_path}/contours/'):
        os.makedirs(f'{mask_path}/contours/')
    
    # Get a list of all image files in the directory with the specified format
    image_files = []
    dfs = []
    for file in os.listdir(mask_path):
        if file.endswith(image_format):
            image_files.append(file)
    image_file_paths = [os.path.join(mask_path, file) for file in image_files]
    for file_path, file_name in zip(image_file_paths, image_files):
        mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

        # display the contours and coordinates in png files
        contour = np.zeros(mask.shape, dtype='uint8')
        marker = np.zeros(mask.shape, dtype='uint8')
        total_coordinates = np.full((2, 2), np.nan)
        for i in range(len(contours)):
            coordinates = np.array(contours[i]).reshape((-1,2))
            if coordinates.shape[0] > minimum_coordinates:
                cv2.drawContours(contour, contours=contours, contourIdx=i, color=(255, 255, 255), thickness=1)
                total_coordinates = np.append(total_coordinates, coordinates, axis=0) 
                for coordinate in coordinates:
                    cv2.drawMarker(marker,coordinate, color=(255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=1, thickness=1)
            
        cv2.imwrite(f'{mask_path}/markers/{file_name[:len(file_name)-len(image_format)]}_markers.png', marker)
        cv2.imwrite(f'{mask_path}/contours/{file_name[:len(file_name)-len(image_format)]}_contours.png', contour)
        total_coordinates = total_coordinates[2:,:] # remove the first two rows of nan
        dfs.append(total_coordinates)
        # make the combined coordinate array
        """
        total_coordinates = np.full((2, 2), np.nan)
        for i in range(len(contours)):
            coordinates = np.array(contours[i]).reshape((-1,2))
            if coordinates.shape[0] > minimum_coordinates:
                #print(coordinates.shape)
                total_coordinates = np.append(total_coordinates, coordinates, axis=0) 
        total_coordinates = total_coordinates[2:,:] # remove the first two rows of nan
        #print(total_coordinates.shape)
        dfs.append(total_coordinates)
        """
    arr = np.array(dfs,dtype=object)
    np.save(f'{mask_path}/vertices.npy', arr)
    savemat(f'{mask_path}/vertices.mat', {'vertices': arr})
    # Save as JSON
    with open(f'{mask_path}/vertices.json', 'w') as f:
        json.dump(dfs, f)
    
def masks2vertices(mask_folder, image_format: str = ".png", minimum_coordinates: int = 100):
    volume_vertices = []
    mat_files = []
    for file in os.listdir(mask_folder):
        if file.endswith('.mat'):
            mask_file = os.path.join(mask_folder, file)
            mat_dict = loadmat(mask_file)
            first_key = list(mat_dict.keys())[3]
            mask_array = mat_dict[first_key]
            #print(mask_array.shape)
            #print(mask_array.dtype)
            #print(type(mask_array))
            if not os.path.exists(f'./{mask_folder}'):  
                os.makedirs(f'./{mask_folder}')
            if os.path.exists(f'./{mask_folder}/{file[:-4]}'):  
                shutil.rmtree(f'./{mask_folder}/{file[:-4]}')
            os.makedirs(f'./{mask_folder}/{file[:-4]}')
            mask_path = f'./{mask_folder}/{file[:-4]}'
            if not os.path.exists(f'{mask_path}/markers/'):
                os.makedirs(f'{mask_path}/markers/')
            if not os.path.exists(f'{mask_path}/contours/'):
                os.makedirs(f'{mask_path}/contours/')
            for i in range(mask_array.shape[2]):
                mask = mask_array[:,:,i]
                file_name = f'{i}.png'
                
                contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                #contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

                # display the contours and coordinates in png files
                if contours:
                    contour = np.zeros(mask.shape, dtype='uint8')
                    marker = np.zeros(mask.shape, dtype='uint8')
                    slice_vertices = np.full((2, 2), np.nan)
                    for i in range(len(contours)):
                        coordinates = np.array(contours[i]).reshape((-1,2))
                        if coordinates.shape[0] > minimum_coordinates:
                            cv2.drawContours(contour, contours=contours, contourIdx=i, color=(255, 255, 255), thickness=1)
                            slice_vertices = np.append(slice_vertices, coordinates, axis=0) 
                            for coordinate in coordinates:
                                cv2.drawMarker(marker,coordinate, color=(255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=1, thickness=1)
                        
                    cv2.imwrite(f'{mask_path}/markers/{file_name}_markers{image_format}', marker)
                    cv2.imwrite(f'{mask_path}/contours/{file_name}_contours{image_format}', contour)
                    slice_vertices = slice_vertices[2:,:] # remove the first two rows of nan
                    volume_vertices.append(slice_vertices.tolist())
                    
            arr = np.array(volume_vertices,dtype=object)
            np.save(f'{mask_path}/vertices.npy', arr)
            savemat(f'{mask_path}/vertices.mat', {'vertices': arr})    
            with open(f'{mask_path}/vertices.json', 'w') as f:
                json.dump(volume_vertices, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert mask to vertices')
    parser.add_argument('--path', '-p', type=str, help='Path to the masks')
    parser.add_argument('--img_fmt', type=str, default='.png', help='Image format')
    parser.add_argument('--min_coord', type=int, default=4, help='Minimum number of coordinates')
    args = parser.parse_args()
    masks2vertices(args.path, args.img_fmt, args.min_coord)