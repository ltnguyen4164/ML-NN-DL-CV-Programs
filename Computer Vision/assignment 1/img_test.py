# Long Nguyen
# 1001705873

# This program is used to test img_transforms.py and create_img_pyramid.py
import img_transforms as it
import create_img_pyramid as cip
# Libraries used to open  and save img files
import os
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import skimage.color as color
from skimage.util import img_as_ubyte

# NOTE: color_space_test.py is recommended to be tested as its own file since it needs command line arguments
# change filename to any image file to test functions
filename = "test.png"
img = io.imread(filename)

if img.ndim == 3 and img.shape[2] == 4:
    img = color.rgba2rgb(img)

# -------- image cropping --------
# change 'size' to any value
size = 100
img_cropped = it.random_crop(img, size)
img_cropped = img_as_ubyte(img_cropped)
# Saved 'img_cropped' to file named 'img_cropped.png'
io.imsave("img_cropped.png", img_cropped)

# -------- patch extraction --------
# change 'num_patches' to any value
num_patches = 4
img_patched = it.extract_patch(img, num_patches)
img_patched = img_as_ubyte(img_patched)
# display the patches compared to the original image (trying to save the patched img proved too difficult)
# plot the original image
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.axis("off")
plt.title("Original Image")

# plot patches
plt.subplot(1, 2, 2)
fig, axes = plt.subplots(num_patches, num_patches, figsize=(10, 10))
for i in range(num_patches):
    for j in range(num_patches):
        axes[i, j].imshow(img_patched[i, j])
        axes[i, j].axis("off")

plt.tight_layout()
plt.show()

# -------- resizing --------
# change 'scale_factor' to any value > 0
scale_factor = 2
img_resized = it.resize_img(img, scale_factor)
img_resized = img_as_ubyte(img_resized)
# resized image is copied to 'img_resized.png'
io.imsave("img_resized.png", img_resized)

# -------- color jitter --------
# change 'hue', 'sat', 'value' to any value, where hue must be (0, 360], and sat and value must be (0,1]
hue, sat, value = 50, 0.2, 0.5
img_jittered = it.color_jitter(img, hue, sat, value)
img_jittered = img_as_ubyte(img_jittered)
# color jittered image is copied to 'img_jittered.png'
io.imsave("img_jittered.png", img_jittered)

# -------- image pyramid --------
height = 4
cip.image_pyramid(img, height)