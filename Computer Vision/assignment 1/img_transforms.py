# Long Nguyen
# 1001705873
import numpy as np
from numpy.lib import stride_tricks
from color_space_test import RGB_to_HSV
from color_space_test import HSV_to_RGB

# function that will generate a random square crop of an image
def random_crop(img, size):
    # get dimensions of the image
    height, width = img.shape[:2]

    # check if crop size is valid
    if size <= 0 or size > min(height, width):
        raise ValueError(f"Crop size must be in the range (0, {min(height, width)}].")
    
    # randomly choose center of crop
    x = np.random.randint(size // 2, width - size // 2 + 1)
    y = np.random.randint(size // 2, height - size // 2 + 1)

    # calculate the crop boundaries
    x_min = x - size // 2
    x_max = x + size // 2
    y_min = y - size // 2
    y_max = y + size // 2

    return img[y_min:y_max, x_min:x_max]
# function that returns n^2 non-overlapping patches
def extract_patch(img, num_patches):
    height, width = img.shape[:2]
    shape = (num_patches, num_patches, height // num_patches, width // num_patches, img.shape[2])

    strides = (
        (height // num_patches) * img.strides[0],
        (width // num_patches) * img.strides[1],
        img.strides[0],                
        img.strides[1],                
        img.strides[2],
    )
    # extract patches
    patches = stride_tricks.as_strided(img, shape=shape, strides=strides)
    return patches
# function that resizes an image
def resize_img(img, factor):
    # check if scale factor is valid
    if factor <= 0:
        raise  ValueError("Scale factor must be a positive integer.")

    # get dimensions of image
    height, width = img.shape[:2]
    new_height, new_width = height // factor, width //factor

    # generate indices for the nearest neighbor interpolation
    row_indices = (np.arange(new_height) * factor).astype(int)
    col_indices = (np.arange(new_width) * factor).astype(int)

    # perform nearest neighbor interpolation
    resized_img = img[np.ix_(row_indices, col_indices)]
    return resized_img
# function which randomly perturbs the HSV values on an image
def color_jitter(img, hue, saturation, value):
    # get random HSV values based on input
    h = np.random.uniform(0, hue)
    s = np.random.uniform(0, saturation)
    v = np.random.uniform(0, value)

    # convert img to hsv, assuming img is in rgb first
    img = img.astype(np.float32) / 255.0
    hsv_image = RGB_to_HSV(img)

    # modify img HSV values with inputted values
    hsv_image[..., 0] = (hsv_image[..., 0] + h / 360.0) % 1.0   # Normalize hue to [0, 1]
    hsv_image[..., 1] = np.clip(hsv_image[..., 1] + s, 0, 1)    # Clamp saturation
    hsv_image[..., 2] = np.clip(hsv_image[..., 2] + v, 0, 1)    # Clamp value

    # convert back to RGB
    rgb_image = HSV_to_RGB(hsv_image)
    return rgb_image