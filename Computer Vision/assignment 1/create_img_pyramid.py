# Long Nguyen
# 1001705873
import os
import numpy as np
from skimage import img_as_ubyte
import skimage.io as io
import skimage.transform as transform

filename = "test.png"

def image_pyramid(img, height):
    h, w = img.shape[:2]

    # save the filename without the extension
    base, ext = os.path.splitext(filename)

    for i in range(1, height):
        scale_factor = 1 / (2 ** i)
        # resize image
        resized_img = transform.rescale(img, scale_factor, anti_aliasing=True, channel_axis=-1, preserve_range=True)
        resized_img = img_as_ubyte(resized_img)

        # save the resized image
        outfile = f"{base}_scale_{2 ** i}x{ext}"
        io.imsave(outfile, resized_img)