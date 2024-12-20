import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import watershed
from skimage.io import imread
from skimage.color import rgb2gray, label2rgb
import scipy.ndimage as nd
from skimage.filters import sobel, threshold_yen
import cv2

IMAGE_NAME = 6
DATA_PATH = 'data/raw/'
DEST_PATH = 'data/segmentation_region/'

image = imread(f'{DATA_PATH}{IMAGE_NAME}.jpeg')
image_gray = rgb2gray(image)
plt.imsave(f"{DEST_PATH}{IMAGE_NAME}_gray.jpeg", image_gray, cmap='gray')

plt.rcParams["figure.figsize"] = (12,8)
elevation_map = sobel(image_gray)
plt.imsave(f"{DEST_PATH}{IMAGE_NAME}_sobel.jpeg", elevation_map, cmap='gray')

thresh = threshold_yen(image_gray)
markers = np.zeros_like(image_gray)
markers[image_gray < thresh] = 1
markers[image_gray >= thresh] = 2 
#Supresion del ruido
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
markers = cv2.morphologyEx(markers.astype(np.uint8), 
                           cv2.MORPH_OPEN,
                           kernel,
                           iterations=2)

plt.imsave(f"{DEST_PATH}{IMAGE_NAME}_markers.jpeg", markers, cmap = 'gray')
 
segmentation = watershed(elevation_map, markers=markers)

plt.imshow(segmentation)
#plt.imsave(f"{DEST_PATH}{IMAGE_NAME}.jpeg", segmentation)
plt.title('Watershed segmentation')
plt.show()
 

