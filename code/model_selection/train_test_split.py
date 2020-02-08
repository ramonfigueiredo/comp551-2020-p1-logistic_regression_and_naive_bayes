import numpy as np


def split_dataset(datasetX, datasetY, trainRatio):
    # Split using scikit-learn
    # Splitting the dataset into the Training set and Test set
    # from sklearn.model_selection import train_test_split
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    # return X_test, X_train, y_test, y_train

    lengthDataset = len(datasetX)
    lengthTrain = int(lengthDataset * trainRatio)

    # Shuffle dataset x and y in the same way .
    combine = np.arange(datasetX.shape[0])
    np.random.shuffle(combine)
    temp_X = datasetX[combine]
    temp_y = datasetY[combine]

    # Split as training and test
    X_train = temp_X[:lengthTrain, :]
    X_test = temp_X[lengthTrain:, :]
    y_train = temp_y[:lengthTrain]
    y_test = temp_y[lengthTrain:]

    return X_train, X_test, y_train, y_test
