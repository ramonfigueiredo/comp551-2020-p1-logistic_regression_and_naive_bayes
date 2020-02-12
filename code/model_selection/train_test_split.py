import numpy as np


def train_test_split(X, y, train_size, shuffle=False):

    length_dataset = len(X)
    length_train = int(length_dataset * train_size)

    # Shuffle dataset x and y in the same way
    if shuffle:
        combine = np.arange(X.shape[0])
        np.random.shuffle(combine)
        X = X[combine]
        y = y[combine]

    # Split as training and test
    X_train = X[:length_train, :]
    X_test = X[length_train:, :]
    y_train = y[:length_train]
    y_test = y[length_train:]

    return X_train, X_test, y_train, y_test
