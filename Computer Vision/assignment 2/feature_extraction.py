import numpy as np
from skimage.feature import hog, SIFT
from skimage.color import rgb2gray
from tqdm import tqdm

def extract_sift(x, y):
    # Code from bag_of_visual_words.ipynb
    sift = SIFT()
    features = []
    y_features = []
    
    # Reshape the data
    img_rgb = x.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    # Convert images to grayscale
    img_gray = rgb2gray(img_rgb)

    for idx in tqdm(range(img_gray.shape[0]), desc="Processing Images"):
        try:
            sift.detect_and_extract(img_gray[idx])
            features.append(sift.descriptors)
            y_features.append(y[idx])
        except:
            pass
    
    return np.array(features, dtype=object), np.array(y_features)

def extract_hog(x):
    features = []

    # Reshape the data
    img_rgb = x.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    # Convert images to grayscale
    img_gray = rgb2gray(img_rgb)
    
    for img in tqdm(img_gray, desc="Processing HOG Features"):
        fd = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
        features.append(fd)
    
    return np.array(features)

if __name__ == "__main__":
    # Load the pre-split data
    data = np.load("cifar10.npz", allow_pickle=True)

    X_train = data["X_train"].astype(np.uint8)
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    
    # Extract features from the training data
    tr_sift_features, tr_y_features = extract_sift(X_train, y_train)
    tr_hog_features = extract_hog(X_train)

    # Extract features from the testing data
    test_sift_features, test_y_features = extract_sift(X_test, y_test)
    test_hog_features = extract_hog(X_test)
    
    # SIFT dictionary
    sift_data = {
        "x_train": tr_sift_features,
        "y_train": tr_y_features,
        "x_test": test_sift_features,
        "y_test": test_y_features
    }

    # HOG dictionary
    hog_data = {
        "x_train": tr_hog_features,
        "y_train": y_train,
        "x_test": test_hog_features,
        "y_test": y_test
    }
    
    # Save the extracted features to a file
    np.savez("cifarSIFT.npz", **sift_data)
    np.savez("cifarHOG.npz", **hog_data)