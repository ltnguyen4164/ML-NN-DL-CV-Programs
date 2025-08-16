# Library to plot images
import matplotlib.pyplot as plt

import numpy as np

# Library used to open images and extract SIFT features
from skimage import io
from skimage.color import rgb2gray, rgba2rgb
from skimage.feature import SIFT, plot_matched_features, match_descriptors

if __name__ == "__main__":
    img1 = io.imread("img/bobbie_template.JPG")
    img2 = io.imread("img/bobbie1.JPG")

    # Extract keypoints from image using SIFT
    sift = SIFT()
    img1_gray = rgb2gray(img1)
    sift.detect_and_extract(img1_gray)
    keypoints1 = sift.keypoints
    descriptors1 = sift.descriptors
    
    img2_gray = rgb2gray(img2)
    sift.detect_and_extract(img2_gray)
    keypoints2 = sift.keypoints
    descriptors2 = sift.descriptors

    matches = match_descriptors(keypoints1, keypoints2)
    
    # Display matches
    fig, ax = plt.subplots()
    plot_matched_features(img1, img2, keypoints0=keypoints1, keypoints1=keypoints2, matches=matches, ax=ax,)
    ax.axis('off')
    plt.show()