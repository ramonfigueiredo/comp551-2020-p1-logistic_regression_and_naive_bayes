from datasets.load_dataset import get_dataset, load_adult
from utils.ml_classifiers_enum import Classifier
from utils.datasets_enum import Datasets
import numpy as np


def run_classifier(classifier, dataset):
    print('\n\nDataset: {}'.format(dataset.name))
    if dataset == Datasets.ADULT:
        X_train, X_test, y_train, y_test = load_adult(load_test_data=True)
        print_data(X_test, X_train, y_test, y_train)
    else:
        X, y = get_dataset(dataset)
        print("\nX:", X)
        print("\ny:", y)

        X_train, X_test, y_train, y_test = split_dataset(X, y, 0.8)
        print_data(X_test, X_train, y_test, y_train)

    print("\n\nFeature scaling:")
    X_train, X_test = feature_scaling(X_test, X_train)
    print("\nX_train:", X_train)
    print("\nX_test:", X_test)

    if classifier == Classifier.LOGISTIC_REGRESSION:
        classifier = fit_logistic_regression(X_train, y_train)
    if classifier == Classifier.NAIVE_BAYES:
        classifier = fit_naive_bayes(X_train, y_train)

    y_pred = predict(X_test, classifier)

    cm = create_confusion_matrix(y_pred, y_test)

    confusion_matrix(cm)

    if dataset == Datasets.ADULT:
        X = np.concatenate((X_train, X_test), axis=0)
        y = np.concatenate((y_train, y_test), axis=0)
        k_fold_cross_validation(X, classifier, y, k=5)
    else:
        k_fold_cross_validation(X, classifier, y, k=5)

    classification_report(y_pred, y_test)

    classification_metrics(y_pred, y_test)


def print_data(X_test, X_train, y_test, y_train):
    print("\nX_train:", X_train)
    print("\nX_test:", X_test)
    print("\ny_train:", y_train)
    print("\ny_test:", y_test)


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

def feature_scaling(X_test, X_train):
    # TODO: Do without use scikit-learn
    # TODO: Change according selected dataset
    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    return X_train, X_test


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
    print("\n\nPredicting the Test set results:\n", y_pred)
    return y_pred


def create_confusion_matrix(y_pred, y_test):
    # TODO: Do without use scikit-learn
    # Creating the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    return cm


def confusion_matrix(cm):
    print("\n\nConfusion Matrix:\n", cm)

    # TODO: Do without use scikit-learn
    # Calculating metrics using the confusion matrix
    TP = cm[0][0]
    FN = cm[0][1]
    TN = cm[1][0]
    FP = cm[1][1]
    print("\nTrue Positive (TP):", TP)
    print("False Negative (FN):", FN)
    print("True Negative (TN):", TN)
    print("False Positive (FP):", FP)


def k_fold_cross_validation(X, classifier, y, k):
    # TODO: Do without use scikit-learn
    # K-fold cross validation
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(classifier, X, y, cv=k)
    print("\n\nK-fold cross validation (k=5). Scores: ", scores)


def classification_report(y_pred, y_test):
    # TODO: Do without use scikit-learn
    print('\n\nClassification report:')
    from sklearn.metrics import classification_report
    print(classification_report(y_test, y_pred))


def classification_metrics(y_pred, y_test):
    # TODO: Do without use scikit-learn
    print("\n\n>>> Classification metrics:")
    from sklearn.metrics import accuracy_score
    print("\n> Accuracy score:", accuracy_score(y_test, y_pred))
    from sklearn.metrics import roc_auc_score
    print("\n> Area Under the Receiver Operating Characteristic Curve (ROC AUC) = ROC AUC Score:",
          roc_auc_score(y_test, y_pred))
    from sklearn.metrics import precision_score
    print("\n> Precision score (average='macro'):", precision_score(y_test, y_pred, average='macro'))
    print("> Precision score (average='micro'):", precision_score(y_test, y_pred, average='micro'))
    print("> Precision score (average='weighted'):", precision_score(y_test, y_pred, average='weighted'))
    print("> Precision score (average=None):", precision_score(y_test, y_pred, average=None))
    from sklearn.metrics import recall_score
    print("\n> Recall score (average='macro'):", recall_score(y_test, y_pred, average='macro'))
    print("> Recall score (average='micro'):", recall_score(y_test, y_pred, average='micro'))
    print("> Recall score (average='weighted'):", recall_score(y_test, y_pred, average='weighted'))
    print("> Recall score (average=None):", recall_score(y_test, y_pred, average=None))
    from sklearn.metrics import f1_score
    print("\n> F1 score (average='macro'):", f1_score(y_test, y_pred, average='macro'))
    print("> F1 score (average='micro'):", f1_score(y_test, y_pred, average='micro'))
    print("> F1 score (average='weighted'):", f1_score(y_test, y_pred, average='weighted'))
    print("> F1 score (average=None):", f1_score(y_test, y_pred, average=None))


if __name__ == '__main__':
    print('\n\n\n==========================')
    print(Classifier.LOGISTIC_REGRESSION.name)
    print('==========================')
    run_classifier(Classifier.LOGISTIC_REGRESSION, Datasets.IONOSPHERE)
    run_classifier(Classifier.LOGISTIC_REGRESSION, Datasets.ADULT)
    run_classifier(Classifier.LOGISTIC_REGRESSION, Datasets.WINE_QUALITY)
    run_classifier(Classifier.LOGISTIC_REGRESSION, Datasets.BREAST_CANCER_DIAGNOSIS)

    print('\n\n\n==========================')
    print(Classifier.NAIVE_BAYES.name)
    print('==========================')
    run_classifier(Classifier.NAIVE_BAYES, Datasets.IONOSPHERE)
    run_classifier(Classifier.NAIVE_BAYES, Datasets.ADULT)
    run_classifier(Classifier.NAIVE_BAYES, Datasets.WINE_QUALITY)
    run_classifier(Classifier.NAIVE_BAYES, Datasets.BREAST_CANCER_DIAGNOSIS)

    print('\n\nDONE!')
