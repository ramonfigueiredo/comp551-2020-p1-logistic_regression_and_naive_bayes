import time

from sklearn.linear_model import LogisticRegression as LR_SkLearn
from sklearn.naive_bayes import GaussianNB as NB_SkLearn

from datasets.load_dataset import get_dataset, load_adult
from linear_model.logistic_regression import LogisticRegression
from linear_model.naive_bayes import GaussianNaiveBayes
from metrics.accuracy_score import evaluate_acc
from model_selection.k_fold_cross_validation import cross_validation
from model_selection.train_test_split import train_test_split
from preprocessing.standard_scaler import feature_scaling
from utils.datasets_enum import Datasets
from utils.ml_classifiers_enum import Classifier
import argparse


def run_classifier(classifier_name, dataset_name):
    X, y = get_dataset(dataset_name)

    X_train, X_test, y_train, y_test = train_test_split(X, y, 0.8)
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
    if classifier_name == Classifier.NAIVE_BAYES_SKLEARN:
        classifier = NB_SkLearn()
    if classifier_name == Classifier.NAIVE_BAYES:
        classifier = GaussianNaiveBayes()

    # Fit the model to the dataset
    classifier.fit(X_train, y_train)

    # k-fold cross validation
    if not (classifier_name == Classifier.LOGISTIC_REGRESSION) and not (classifier_name == Classifier.NAIVE_BAYES):
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
    scores, fit_times, predict_times, model_accuracy_times, processing_times_of_folds = \
        cross_validation(classifier, X, y, k, randomize=True, verbose=False)

    print("\n\nK-fold cross validation (k=5). Scores: ", scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    print("\n\nK-fold cross validation (k=5). Fit times in seconds: ", fit_times)
    print("Fit time (seconds): %0.2f (+/- %0.2f)" % (fit_times.mean(), fit_times.std() * 2))

    print("\n\nK-fold cross validation (k=5). Predict times in seconds: ", predict_times)
    print("Predict time (seconds): %0.2f (+/- %0.2f)" % (predict_times.mean(), predict_times.std() * 2))

    print("\n\nK-fold cross validation (k=5). Model accuracy calculation times in seconds: ", model_accuracy_times)
    print("Model accuracy calculation time (seconds): %0.2f (+/- %0.2f)" % (
    model_accuracy_times.mean(), model_accuracy_times.std() * 2))

    print("\n\nK-fold cross validation (k=5). Processing times of folds in seconds: ", processing_times_of_folds)
    print("Processing time of fold (seconds): %0.2f (+/- %0.2f)" % (
    processing_times_of_folds.mean(), processing_times_of_folds.std() * 2))


def classification_report(y_pred, y_test):
    print('\n\nClassification report:')
    from sklearn.metrics import classification_report
    print(classification_report(y_test, y_pred))


def classification_metrics(y_pred, y_test):
    print("\n\n>>> Classification metrics:")
    print("\n> Accuracy score:", evaluate_acc(y_test, y_pred))
    from sklearn.metrics import roc_auc_score
    # print("\n> Area Under the Receiver Operating Characteristic Curve (ROC AUC) = ROC AUC Score:",
    #       roc_auc_score(y_test, y_pred))
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


def run_classifier_given_dataset(classifier):
    print('\n\n\n==========================')
    print(classifier.name)
    print('==========================')
    if options.dataset.upper() == Datasets.IONOSPHERE.name or options.dataset.lower() == 'i':
        run_classifier(classifier, Datasets.IONOSPHERE)
    elif options.dataset.upper() == Datasets.ADULT.name or options.dataset.lower() == 'a':
        run_classifier(classifier, Datasets.ADULT)
    elif options.dataset.upper() == Datasets.WINE_QUALITY.name or options.dataset.lower() == 'wq':
        run_classifier(classifier, Datasets.WINE_QUALITY)
    elif options.dataset.upper() == Datasets.BREAST_CANCER_DIAGNOSIS.name or options.dataset.lower() == 'bcd':
        run_classifier(classifier, Datasets.BREAST_CANCER_DIAGNOSIS)


if __name__ == '__main__':
    start = time.time()

    parser = argparse.ArgumentParser(description='MiniProject 1: Logistic Regression and Naive Bayes. Authors: Ramon Figueiredo Pessoa, Rafael Gomes Braga, Ege Odaci',
                                     epilog='COMP 551 (001/002), Applied Machine Learning, Winter 2020, McGill University.')

    parser.add_argument('-c', '--classifier', action='store', dest='classifier',
                        help='Classifier used '
                             '(Options: all, '
                             'logistic_regression_sklearn OR lrskl, '
                             'logistic_regression OR lr, '
                             'naive_bayes_sklearn OR nbskl'
                             'naive_bayes OR nb).',
                        default='all')

    parser.add_argument('-d', '--dataset', action='store', dest='dataset',
                        help='Database used '
                             '(Options: all, '
                             'ionosphere OR i '
                             'adult OR a '
                             'wine_quality OR wq'
                             'breast_cancer_diagnosis OR bcd).',
                        default='all')

    parser.add_argument('-v', '--version', action='version', version='%(prog)s 1.0')

    options = parser.parse_args()

    print(parser.description, '\nRunning with options: ')
    print('\tclassifier =', options.classifier.upper())
    print('\tdataset =', options.dataset.upper())

    if options.classifier.upper() == Classifier.LOGISTIC_REGRESSION_SKLEARN.name or options.classifier.lower() == 'lrskl':
        run_classifier_given_dataset(Classifier.LOGISTIC_REGRESSION_SKLEARN)
    elif options.classifier.upper() == Classifier.LOGISTIC_REGRESSION.name or options.classifier.lower() == 'lr':
        run_classifier_given_dataset(Classifier.LOGISTIC_REGRESSION)
    elif options.classifier.upper() == Classifier.NAIVE_BAYES_SKLEARN.name or options.classifier.lower() == 'nbskl':
        run_classifier_given_dataset(Classifier.NAIVE_BAYES_SKLEARN)
    elif options.classifier.upper() == Classifier.NAIVE_BAYES.name or options.classifier.lower() == 'nb':
        run_classifier_given_dataset(Classifier.NAIVE_BAYES)
    else:
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
        run_classifier(Classifier.LOGISTIC_REGRESSION, Datasets.IONOSPHERE)
        run_classifier(Classifier.LOGISTIC_REGRESSION, Datasets.ADULT)
        run_classifier(Classifier.LOGISTIC_REGRESSION, Datasets.WINE_QUALITY)
        run_classifier(Classifier.LOGISTIC_REGRESSION, Datasets.BREAST_CANCER_DIAGNOSIS)

        print('\n\n\n==========================')
        print(Classifier.NAIVE_BAYES_SKLEARN.name)
        print('==========================')
        run_classifier(Classifier.NAIVE_BAYES_SKLEARN, Datasets.IONOSPHERE)
        run_classifier(Classifier.NAIVE_BAYES_SKLEARN, Datasets.ADULT)
        run_classifier(Classifier.NAIVE_BAYES_SKLEARN, Datasets.WINE_QUALITY)
        run_classifier(Classifier.NAIVE_BAYES_SKLEARN, Datasets.BREAST_CANCER_DIAGNOSIS)

        print('\n\n\n==========================')
        print(Classifier.NAIVE_BAYES.name)
        print('==========================')
        run_classifier(Classifier.NAIVE_BAYES, Datasets.IONOSPHERE)
        run_classifier(Classifier.NAIVE_BAYES, Datasets.ADULT)
        run_classifier(Classifier.NAIVE_BAYES, Datasets.WINE_QUALITY)
        run_classifier(Classifier.NAIVE_BAYES, Datasets.BREAST_CANCER_DIAGNOSIS)

    print('\n\nDONE!')
    print('It took', time.time() - start, 'seconds.')
