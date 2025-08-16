# Long Nguyen
# 1001705873
import numpy as np
from skimage import img_as_ubyte
import skimage.io as io
import skimage.color as color
import sys

# function to convert RGB to HSV
def RGB_to_HSV(img):
    # get RGB values
    R = img[..., 0]
    G = img[..., 1]
    B = img[..., 2]
    
    # calculate V value
    V = np.max(img, axis=2)
    # calculate chroma value
    C = V - np.min(img, axis=2)
    # calculate saturation
    with np.errstate(divide='ignore', invalid='ignore'):
        S = np.where(V != 0, C / V, 0)

    # calculate h' values
    with np.errstate(divide='ignore', invalid='ignore'):
        h_prime = np.where(C == 0, 0,
                           np.where(V == R, ((G - B) / C) % 6,
                                    np.where(V == G, ((B - R) / C) + 2, ((R - G) / C) + 4)))

    # calculate final hue
    H = 60 * h_prime
    H = np.where(H < 0, H + 360, H)
    H = H / 360.0

    # combine H, S, and V into the output HSV image
    HSV = np.stack([H, S, V], axis=-1)
    return HSV
# function to convert HSV to RGB
def HSV_to_RGB(img):
    # get HSV values
    H = img[..., 0]
    S = img[..., 1]
    V = img[..., 2]
    
    # calculate chroma value
    C = V * S

    # scale hue to range [0, 6]
    H *= 6
    
    # calculate X
    X = C * (1 - np.abs((H % 2) - 1))

    # initialize RGB values
    R = np.zeros_like(H)
    G = np.zeros_like(H)
    B = np.zeros_like(H)

    # assign RGB values based on H
    R = np.where((0 <= H) & (H < 1), C, R)
    G = np.where((0 <= H) & (H < 1), X, G)

    R = np.where((1 <= H) & (H < 2), X, R)
    G = np.where((1 <= H) & (H < 2), C, G)

    G = np.where((2 <= H) & (H < 3), C, G)
    B = np.where((2 <= H) & (H < 3), X, B)

    G = np.where((3 <= H) & (H < 4), X, G)
    B = np.where((3 <= H) & (H < 4), C, B)

    R = np.where((4 <= H) & (H < 5), X, R)
    B = np.where((4 <= H) & (H < 5), C, B)

    R = np.where((5 <= H) & (H < 6), C, R)
    B = np.where((5 <= H) & (H < 6), X, B)
    
    # add offset
    m = V - C
    R = R + m
    G = G + m
    B = B + m

    RGB = np.stack([R, G, B], axis=-1)
    return RGB

if __name__ == "__main__":    
    if len(sys.argv) != 5:
        print("Usage: python script.py <filename> <hue_mod> <sat_mod> <val_mod>")
        sys.exit(1)

    filename = sys.argv[1]
    hue = float(sys.argv[2])
    sat = float(sys.argv[3])
    val = float(sys.argv[4])

    # check if input values are within range
    if not (0 <= hue <= 360):
        print("Error: Hue modification must be in the range [0, 360].")
        sys.exit(1)
    if not (0 <= sat <= 1) or not (0 <= val <= 1):
        print("Error: Saturation and value modifications must be in the range [0, 1].")
        sys.exit(1)

    # load image file
    img = io.imread(filename)
    if img.ndim == 3 and img.shape[2] == 4:
        img = color.rgba2rgb(img)

    # convert img to float in [0, 1]
    img = img.astype(np.float32) / 255.0
    hsv_image = RGB_to_HSV(img)

    # modify img HSV values with inputted values
    hsv_image[..., 0] = (hsv_image[..., 0] + hue / 360.0) % 1.0   # Normalize hue to [0, 1]
    hsv_image[..., 1] = np.clip(hsv_image[..., 1] + sat, 0, 1)    # Clamp saturation
    hsv_image[..., 2] = np.clip(hsv_image[..., 2] + val, 0, 1)    # Clamp value

    # convert back to RGB
    rgb_image = HSV_to_RGB(hsv_image)
    rgb_image = img_as_ubyte(rgb_image)

    # save img to another file
    io.imsave("outfile.png", rgb_image)