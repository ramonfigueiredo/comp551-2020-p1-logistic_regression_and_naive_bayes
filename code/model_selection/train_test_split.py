import numpy as np


def train_test_split(X, y, train_size):

    length_dataset = len(X)
    length_train = int(length_dataset * train_size)

    # Shuffle dataset x and y in the same way
    combine = np.arange(X.shape[0])
    np.random.shuffle(combine)
    temp_X = X[combine]
    temp_y = y[combine]

    # Split as training and test
    X_train = temp_X[:length_train, :]
    X_test = temp_X[length_train:, :]
    y_train = temp_y[:length_train]
    y_test = temp_y[length_train:]

    return X_train, X_test, y_train, y_test
