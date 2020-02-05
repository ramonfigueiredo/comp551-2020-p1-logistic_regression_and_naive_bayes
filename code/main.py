from datasets.load_dataset import get_dataset, Datasets
import numpy as np


def run_logistic_regression(dataset):
    print('Dataset: {}'.format(dataset.name))
    X, y = get_dataset(dataset)
    print(X, y)

    X_test, X_train, y_test, y_train = split_dataset(X, y, 0.8)
    print(X_test)
    print(X_train)
    X_test, X_train = feature_scaling(X_test, X_train)

    classifier = fit_logistic_regression(X_train, y_train)

    y_pred = predict(X_test, classifier)

    cm = create_confusion_matrix(y_pred, y_test)

    calculate_metrics(cm)

    k_fold_cross_validation(X, classifier, y, k=5)

    calculate_model_accuracy(y_pred, y_test)


def run_naive_bayes(dataset):
    print('Dataset: {}'.format(dataset.name))
    X, y = get_dataset(dataset)
    print(X, y)

    X_test, X_train, y_test, y_train = split_dataset(X, y, 0.8)

    X_test, X_train = feature_scaling(X_test, X_train)

    classifier = fit_naive_bayes(X_train, y_train)

    y_pred = predict(X_test, classifier)

    cm = create_confusion_matrix(y_pred, y_test)

    calculate_metrics(cm)

    k_fold_cross_validation(X, classifier, y, k=5)

    calculate_model_accuracy(y_pred, y_test)
#
# def split_data(datasetX,datasetY, trainRatio):
#     lengthDataset = len(datasetX)
#     lengthTrain = int(lengthDataset * trainRatio)
#
# #Shuffle dataset x and y in the same way .
#     combine = np.arrange(datasetX.shape[0])
#     np.random.shuffle(combine)
#     tempx = datasetX[combine]
#     tempy = datasetY[combine]
#     #Seperate as training and test
#     x_train = tempx[:lengthTrain, :]
#     x_test = tempx[lengthTrain:, :]
#     y_train = tempy[:lengthTrain, :]
#     y_test = tempy[:lengthTrain, :]

def split_dataset(datasetX, datasetY, trainRatio):
    # TODO: Do without use scikit-learn
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

    return X_test, X_train, y_test, y_train

def feature_scaling(X_test, X_train):
    # TODO: Do without use scikit-learn
    # TODO: Change according selected dataset
    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_test, X_train


def fit_logistic_regression(X_train, y_train):
    # TODO: Do without use scikit-learn
    # Fitting Logistic Regression to the Training set
    from sklearn.linear_model import LogisticRegression
    # How to fix non-convergence in LogisticRegressionCV
    # https://stats.stackexchange.com/questions/184017/how-to-fix-non-convergence-in-logisticregressioncv
    classifier = LogisticRegression(random_state=0, max_iter=1000)
    classifier.fit(X_train, y_train)
    return classifier


def fit_naive_bayes(X_train, y_train):
    # TODO: Do without use scikit-learn
    # Fitting Naive Bayes to the Training set
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    return classifier


def predict(X_test, classifier):
    # TODO: Do without use scikit-learn
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    print("Predicting the Test set results\n", y_pred)
    return y_pred


def create_confusion_matrix(y_pred, y_test):
    # TODO: Do without use scikit-learn
    # Creating the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    return cm


def calculate_metrics(cm):
    print("Confusion Matrix\n", cm)

    # TODO: Do without use scikit-learn
    # Calculating metrics using the confusion matrix
    TP = cm[0][0]
    FN = cm[0][1]
    TN = cm[1][0]
    FP = cm[1][1]
    print("True Positive (TP):", TP)
    print("False Negative (FN):", FN)
    print("True Negative (TN):", TN)
    print("False Positive (FP):", FP)
    print("\n")

    if (TP + TN + FP + FN) > 0:
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        print("Accuracy = (TP + TN) / (TP + TN + FP + FN): %.2f %%" % (accuracy * 100))

    if (TP + FN) > 0:
        recall = TP / (TP + FN)
        print("Recall = TP / (TP + FN): %.2f %%" % (recall * 100))

    if (TP + FP) > 0:
        precision = TP / (TP + FP)
        print("Precision = TP / (TP + FP): %.2f %%" % (precision * 100))

    if (recall + precision) > 0:
        Fmeasure = (2 * recall * precision) / (recall + precision)
        print("Fmeasure = (2 * recall * precision) / (recall + precision): %.2f %%" % (Fmeasure * 100))


def k_fold_cross_validation(X, classifier, y, k):
    # TODO: Do without use scikit-learn
    # K-fold cross validation
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(classifier, X, y, cv=k)
    print("K-fold cross validation (k=5). Scores: ", scores)


def calculate_model_accuracy(y_pred, y_test):
    # TODO: Do without use scikit-learn
    # Model accuracy
    from sklearn.metrics import accuracy_score
    model_accuracy = accuracy_score(y_test, y_pred)
    print("Model accuracy (accuracy score): ", model_accuracy)


if __name__ == '__main__':
    print('\n\n==> Logistic Regression')
    run_logistic_regression(Datasets.IONOSPHERE)
    # run_logistic_regression(Datasets.ADULT)
    run_logistic_regression(Datasets.WINE_QUALITY)
    run_logistic_regression(Datasets.BREAST_CANCER_DIAGNOSIS)

    print('\n\n==> Naive Bayes')
    run_naive_bayes(Datasets.IONOSPHERE)
    # run_naive_bayes(Datasets.ADULT)
    run_naive_bayes(Datasets.WINE_QUALITY)
    run_naive_bayes(Datasets.BREAST_CANCER_DIAGNOSIS)

    print('DONE!')
