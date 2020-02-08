import time

from sklearn.linear_model import LogisticRegression as LR_SkLearn
from sklearn.naive_bayes import GaussianNB as NB_SkLearn

from datasets.load_dataset import get_dataset, load_adult
from linear_model.logistic_regression import LogisticRegression
from metrics.accuracy_score import accuracy_score
from model_selection.train_test_split import split_dataset
from preprocessing.standard_scaler import feature_scaling
from utils.datasets_enum import Datasets
from utils.ml_classifiers_enum import Classifier


def run_classifier(classifier_name, dataset):
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
    X_train, X_test = feature_scaling(X_train, X_test)
    print("\nX_train:", X_train)
    print("\nX_test:", X_test)

    if classifier_name == Classifier.LOGISTIC_REGRESSION_SKLEARN:
        # How to fix non-convergence in LogisticRegressionCV
        # https://stats.stackexchange.com/questions/184017/how-to-fix-non-convergence-in-logisticregressioncv
        classifier = LR_SkLearn(random_state=0, max_iter=1000)
    if classifier_name == Classifier.LOGISTIC_REGRESSION:
        classifier = LogisticRegression()
    if classifier_name == Classifier.NAIVE_BAYES:
        # TODO: Do without use scikit-learn
        classifier = NB_SkLearn()

    # Fit the model to the dataset
    classifier.fit(X_train, y_train)

    # k-fold cross validation
    if not (classifier_name == Classifier.LOGISTIC_REGRESSION):
        k_fold_cross_validation(X_train, classifier, y_train, k=5)

    # Predict the labels
    y_pred = classifier.predict(X_test)
    print("\n\nPredicting the Test set results:\n", y_pred)

    cm = create_confusion_matrix(y_pred, y_test)

    print_confusion_matrix(cm)

    classification_report(y_pred, y_test)

    classification_metrics(y_pred, y_test)


def print_data(X_test, X_train, y_test, y_train):
    print("\nX_train:", X_train)
    print("\nX_test:", X_test)
    print("\ny_train:", y_train)
    print("\ny_test:", y_test)


def create_confusion_matrix(y_pred, y_test):
    # TODO: Do without use scikit-learn
    # Creating the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    return cm


def print_confusion_matrix(cm):
    # https://en.wikipedia.org/wiki/Confusion_matrix
    print("\n\nConfusion Matrix:\n", cm)

    # TP, FP, FN, TN
    TP = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TN = cm[1][1]
    print("\nTrue Positive (TP):", TP)
    print("False Positive (FP):", FP)
    print("False Negative (FN):", FN)
    print("True Negative (TN):", TN)


def k_fold_cross_validation(X, classifier, y, k):
    # TODO: Do without use scikit-learn
    # K-fold cross validation
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(classifier, X, y, cv=k)
    print("\n\nK-fold cross validation (k=5). Scores: ", scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


def classification_report(y_pred, y_test):
    # TODO: Do without use scikit-learn
    print('\n\nClassification report:')
    from sklearn.metrics import classification_report
    print(classification_report(y_test, y_pred))


def classification_metrics(y_pred, y_test):
    # TODO: Do without use scikit-learn
    print("\n\n>>> Classification metrics:")
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
    start = time.time()

    print('\n\n\n==========================')
    print(Classifier.LOGISTIC_REGRESSION_SKLEARN.name)
    print('==========================')
    run_classifier(Classifier.LOGISTIC_REGRESSION_SKLEARN, Datasets.IONOSPHERE)
    run_classifier(Classifier.LOGISTIC_REGRESSION_SKLEARN, Datasets.ADULT)
    run_classifier(Classifier.LOGISTIC_REGRESSION_SKLEARN, Datasets.WINE_QUALITY)
    run_classifier(Classifier.LOGISTIC_REGRESSION_SKLEARN, Datasets.BREAST_CANCER_DIAGNOSIS)

    print('\n\n\n==========================')
    print(Classifier.LOGISTIC_REGRESSION.name)
    print('==========================')
    # run_classifier(Classifier.LOGISTIC_REGRESSION, Datasets.IONOSPHERE)
    # run_classifier(Classifier.LOGISTIC_REGRESSION, Datasets.ADULT)
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
    print('It took', time.time() - start, 'seconds.')
