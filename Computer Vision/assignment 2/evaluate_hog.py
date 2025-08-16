import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

# Load HOG feature data
data = np.load("cifarHOG.npz", allow_pickle=True)

Xd_train = data["x_train"]  # HOG features for training
yd_train = data["y_train"]
Xd_test = data["x_test"]
yd_test = data["y_test"]

# Concatenate all HOG feature vectors into one array for clustering
hog_features_np = np.vstack(Xd_train)

# Define vocabulary size for KMeans clustering
vocab_size = 100
kmeans = KMeans(n_clusters=vocab_size)

# Fit KMeans on the extracted HOG features
kmeans.fit(hog_features_np)

# Create histograms for each image based on assigned clusters
img_histogram = []

for feature in tqdm(Xd_train, desc="Processing HOG Histograms"):
    clusters = kmeans.predict(feature.reshape(1, -1))
    histogram, _ = np.histogram(clusters, bins=vocab_size, range=(0, vocab_size))
    img_histogram.append(histogram)

# Convert histograms to a NumPy array
img_histogram_np = np.array(img_histogram)

# Apply TF-IDF transformation
tfidf = TfidfTransformer()
tfidf.fit(img_histogram_np)
img_histogram_tfidf = tfidf.transform(img_histogram_np)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    img_histogram_tfidf, np.array(yd_train, dtype=int), test_size=0.2, random_state=42
)

# Train SVM classifier
svm = LinearSVC()
svm.fit(X_train, y_train)
# Evaluate the model
accuracy = svm.score(X_test, y_test)
print(f"SVM accuracy on HOG features: {accuracy:.2f}")
# Get number of correct classifications
num_correct = accuracy * len(y_test)
print(f"Number of correctly classified images: {int(num_correct)}")