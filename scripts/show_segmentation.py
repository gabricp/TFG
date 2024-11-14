
import argparse
import matplotlib.pyplot as plt
import matplotlib
from skimage.io import imread
from skimage.transform import resize
from skimage.color import  label2rgb, rgb2gray
import scipy.ndimage as nd
import numpy as np
from skimage.filters import try_all_threshold, threshold_mean
from segmentation_mask_overlay import overlay_masks


IMAGES = [6,9,23,31]
METHOD = 'region'
SAVE = True

if METHOD not in ['conv', 'region']:
    raise ValueError('Method must be either "conv" or "region"')

data_seg = "data/segmentation/" if METHOD == 'conv' else "data/segmentation_region/"
data_path = "data/raw/"
images = []
segmentations = []
for image in IMAGES:
    try:
        im = imread(f'{data_path}{image}.jpeg')
        im = resize(im, (1440, 1080))
        segmentation = imread(f'{data_seg}{image}.jpeg')
        segmentation = resize(segmentation, (1440, 1080))
    except Exception as e:
        print(f"Error loading image {image}")
        continue
    images.append(im)
    segmentations.append(segmentation)

fig, axs = plt.subplots(2, len(images), figsize=(20, 10))
for i, (image, segmentation) in enumerate(zip(images, segmentations)):
    axs[0, i].imshow(image)
    axs[0, i].set_title(f'Original')
    axs[0, i].axis('off')
    axs[1, i].imshow(segmentation)
    axs[1, i].set_title(f'Segmented')
    axs[1, i].axis('off')
if SAVE:
    plt.savefig(f"{data_seg}results.jpeg")
plt.show()