# Long Nguyen
# 1001705873

import numpy as np

def generate_data(n_samples=200, std_dev=2.5, class_distance=0.9, train_test_shift=2.2, seed=42):
    np.random.seed(seed)
    
    # Generate random means for both classes (ensuring their distance is <= 1)
    mean_class1_train = np.random.uniform(-0.5, 0.5, size=2)
    mean_class2_train = mean_class1_train + np.random.uniform(-class_distance, class_distance, size=2)
    
    mean_class1_test = mean_class1_train + np.random.uniform(-train_test_shift, train_test_shift, size=2)
    mean_class2_test = mean_class2_train + np.random.uniform(-train_test_shift, train_test_shift, size=2)

    # Generate training data
    class1_train = np.random.normal(loc=mean_class1_train, scale=std_dev, size=(n_samples, 2))
    class2_train = np.random.normal(loc=mean_class2_train, scale=std_dev, size=(n_samples, 2))
    
    # Generate test data
    class1_test = np.random.normal(loc=mean_class1_test, scale=std_dev, size=(n_samples, 2))
    class2_test = np.random.normal(loc=mean_class2_test, scale=std_dev, size=(n_samples, 2))

    # Create labels (0 for class 1, 1 for class 2)
    y_train = np.array([0] * n_samples + [1] * n_samples)
    y_test = np.array([0] * n_samples + [1] * n_samples)

    # Stack the dataset
    X_train = np.vstack((class1_train, class2_train))
    X_test = np.vstack((class1_test, class2_test))

    # Shuffle the dataset
    train_indices = np.random.permutation(len(X_train))
    test_indices = np.random.permutation(len(X_test))

    X_train, y_train = X_train[train_indices], y_train[train_indices]
    X_test, y_test = X_test[test_indices], y_test[test_indices]

    return X_train, y_train, X_test, y_test

# Generate the dataset
train_data, train_labels, test_data, test_labels = generate_data()

# Save the dataset
np.savetxt('ltn_training.txt', np.hstack((train_data, train_labels.reshape(-1, 1))), delimiter=" ", comments="", fmt=["%.6f", "%.6f", "%d"])
np.savetxt('ltn_test.txt', np.hstack((test_data, test_labels.reshape(-1, 1))), delimiter=" ", comments="", fmt=["%.6f", "%.6f", "%d"])