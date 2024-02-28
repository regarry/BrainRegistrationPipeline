import cv2
import os
import numpy as np
import shutil
from scipy.io import savemat, loadmat  
import mat73
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
    
def masks2vertices(mask_folder, image_format: str = ".png", minimum_coordinates: int = 3):
    
    for file in os.listdir(mask_folder):
        if file.endswith('.mat'):
            mask_file = os.path.join(mask_folder, file)
            try:
                mat_dict = mat73.loadmat(mask_file)
                #print(mat_dict.keys())
                first_key = list(mat_dict.keys())[0]
            except:
                mat_dict = loadmat(mask_file)
                first_key = list(mat_dict.keys())[3]
            
            mask_array = mat_dict[first_key].astype('uint8') 
            volume_vertices = []
            number_of_valid_slices = 0
            volume_vertices_dict = {"0": mask_array.shape[:1]}
            #print(volume_vertices_dict)
            print(mask_array.shape)
            print(mask_array.dtype)
            print(type(mask_array))
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
                #print(mask[256,256])
                #print(mask.dtype)
                
                contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                #contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
                #print({"1":contours})
                # display the contours and coordinates in png files
                slice_vertices_list = []
                if contours:
                    contour_img = np.zeros(mask.shape, dtype='uint8')
                    marker_img = np.zeros(mask.shape, dtype='uint8')
                    #slice_vertices = np.full((2, 2), np.nan)

                    
                    slice_vertices = np.array(contours[0]).reshape((-1,2))
                    #print(f"{slice_vertices.shape}")
                    # removed for i in range(len(contours)): # limited to one contour per slice for now
                    if slice_vertices.shape[0] > minimum_coordinates:
                        cv2.drawContours(contour_img, contours=contours, contourIdx=0, color=(255, 255, 255), thickness=1) # only did index 0
                        #slice_vertices = np.append(slice_vertices, coordinates, axis=0) 
                        for pair in slice_vertices:
                            slice_vertices_list.append([pair.tolist()])
                            
                            cv2.drawMarker(marker_img, pair, color=(255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=1, thickness=1)
                        #print({f"{i}": slice_vertices_list})
                        volume_vertices_dict.update({f"{i}": slice_vertices_list})
                        number_of_valid_slices += 1
                        # need to think more about k because I need to be able to backtrack to the original slice
                        cv2.imwrite(f'{mask_path}/markers/{i}_markers{image_format}', marker_img)
                        cv2.imwrite(f'{mask_path}/contours/{i}_contours{image_format}', contour_img)
                    #slice_vertices = slice_vertices[2:,:] # remove the first two rows of nan
                    #volume_vertices.append(slice_vertices.tolist())
            
            # may need to change the size of the third dimension to match the number of slices with valid contours        
            #arr = np.array(volume_vertices,dtype=object)
            #np.save(f'{mask_path}/vertices.npy', arr)
            #savemat(f'{mask_path}/vertices.mat', {'vertices': arr})
            volume_size = [mask_array.shape[0], mask_array.shape[1], number_of_valid_slices]
            volume_vertices_dict.update({"0": volume_size})  
            with open(f'{mask_path}/vertices.json', 'w') as f:
                json.dump(volume_vertices_dict, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert mask to vertices')
    parser.add_argument('--path', '-p', type=str, help='Path to the masks')
    parser.add_argument('--img_fmt', type=str, default='.png', help='Image format')
    parser.add_argument('--min_coord', type=int, default=3, help='Minimum number of coordinates')
    args = parser.parse_args()
    masks2vertices(args.path, args.img_fmt, args.min_coord)