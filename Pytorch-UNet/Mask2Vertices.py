from pandas import DataFrame
import cv2
import scipy.ndimage as ndi
import numpy as np
import matplotlib.pyplot as plt

real_image = cv2.imread('cfos_049.png')
#print(np.unique(real_image))
#print(real_image.shape)
cv2.imshow('real image',real_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

mask = cv2.imread('out_1b_049.png', cv2.IMREAD_GRAYSCALE)
#print(np.unique(mask))

# to remove random outliers from the model that are not part of the mask
"""
blur = cv2.medianBlur(image, 5)
cv2.imshow('median filter',blur)
cv2.waitKey(0)
cv2.destroyAllWindows()

edged = cv2.Canny(blur, 125, 126) 
cv2.imshow('Canny',edged)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#print(contours)

blank = np.zeros(mask.shape, dtype='uint8')
cv2.drawContours(blank, contours=contours, contourIdx=0, color=(255, 255, 255), thickness=1)
cv2.imwrite('contours.png', blank)
cv2.imshow('Contours', blank)
cv2.waitKey(0)
cv2.destroyAllWindows()

blank = np.zeros(mask.shape, dtype='uint8')
coordinates = np.array(contours[0]).reshape((-1,2))
for coordinate in coordinates:
    cv2.drawMarker(blank,coordinate, color=(255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=1, thickness=1)
cv2.imwrite('markers.png', blank)
cv2.imshow('Markers', blank)
cv2.waitKey(0)
cv2.destroyAllWindows()