# Library to plot images
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

import numpy as np

# Library used to open images and extract SIFT features
from skimage import io
from skimage.color import rgb2gray, rgba2rgb
from skimage.feature import SIFT

def keypoint_matching(keypoints1, keypoints2):
    matches = []
    
    # Brute force keypoint matching using distances
    for i, kp1 in enumerate(keypoints1):
        # Compute distances
        distances = np.linalg.norm(keypoints2 - kp1, axis=1)
        # Find closest keypoint in keypoint2
        best_indx = np.argmin(distances)

        matches.append([i, best_indx])

    return np.array(matches, dtype=int)

def plot_matches(img1, img2, keypoints1, keypoints2, matches):
    # Code from ransac.ipynb
    dst = keypoints1[matches[:, 0]]
    src = keypoints2[matches[:, 1]]
    
    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.imshow(img1, cmap='gray')
    ax2.imshow(img2, cmap='gray')
    
    for i in range(src.shape[0]):
        coordB = [dst[i, 1], dst[i, 0]]
        coordA = [src[i, 1], src[i, 0]]
        con = ConnectionPatch(xyA=coordA, xyB=coordB, coordsA="data", coordsB="data", axesA=ax2, axesB=ax1, color="red")
        ax2.add_artist(con)
        ax1.plot(dst[i, 1], dst[i, 0], 'ro')
        ax2.plot(src[i, 1], src[i, 0], 'ro')
    
    plt.show()

if __name__ == "__main__":
    img1 = io.imread("img/bobbie_template.JPG")
    img2 = io.imread("img/bobbie1.JPG")

    # Extract keypoints from image using SIFT
    sift = SIFT()
    img1_gray = rgb2gray(img1)
    sift.detect_and_extract(img1_gray)
    keypoints1 = sift.keypoints
    
    img2_gray = rgb2gray(img2)
    sift.detect_and_extract(img2_gray)
    keypoints2 = sift.keypoints

    matches = keypoint_matching(keypoints1, keypoints2)
    
    # Display matches
    plot_matches(img1, img2, keypoints1, keypoints2, matches)