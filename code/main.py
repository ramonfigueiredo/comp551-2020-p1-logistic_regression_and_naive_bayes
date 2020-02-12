import argparse
import logging
import os
import time

from sklearn.linear_model import LogisticRegression as LR_SkLearn
from sklearn.naive_bayes import GaussianNB as NB_SkLearn

from datasets.load_dataset import get_dataset
from linear_model.logistic_regression import LogisticRegression
from linear_model.naive_bayes import GaussianNaiveBayes
from metrics.accuracy_score import evaluate_acc
from model_selection.k_fold_cross_validation import cross_validation
from model_selection.train_test_split import train_test_split
from plotting.cost_vs_iterations import cost_vs_iterations_plotting
from preprocessing.standard_scaler import feature_scaling
from utils.datasets_enum import Datasets
from utils.ml_classifiers_enum import Classifier
from plotting.heatmap_plotting import heatmap_plotting


def run_classifier(classifier_name, dataset_name, training_set_size):
    X, y = get_dataset(dataset_name)

    print("\n\nFeature scaling:")
    X = feature_scaling(X)
    print("\nX:", X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, training_set_size, shuffle=True)
    print_data(X_test, X_train, y_test, y_train)

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


def run_classifier_given_dataset(classifier, training_set_size):
    print('\n\n\n==========================')
    print(classifier.name)
    print('==========================')
    if options.dataset.upper() == Datasets.IONOSPHERE.name or options.dataset.lower() == 'i':
        run_classifier(classifier, Datasets.IONOSPHERE, training_set_size)
    elif options.dataset.upper() == Datasets.ADULT.name or options.dataset.lower() == 'a':
        run_classifier(classifier, Datasets.ADULT, training_set_size)
    elif options.dataset.upper() == Datasets.WINE_QUALITY.name or options.dataset.lower() == 'wq':
        run_classifier(classifier, Datasets.WINE_QUALITY, training_set_size)
    elif options.dataset.upper() == Datasets.BREAST_CANCER_DIAGNOSIS.name or options.dataset.lower() == 'bcd':
        run_classifier(classifier, Datasets.BREAST_CANCER_DIAGNOSIS, training_set_size)


if __name__ == '__main__':
    start = time.time()

    parser = argparse.ArgumentParser(description='MiniProject 1: Logistic Regression and Naive Bayes. Authors: Ramon Figueiredo Pessoa, Rafael Gomes Braga, Ege Odaci',
                                     epilog='COMP 551 (001/002), Applied Machine Learning, Winter 2020, McGill University.')

    parser.add_argument('-c', '--classifier', action='store', dest='classifier',
                        help='Classifier used '
                             '(Options: all, '
                             'logistic_regression_sklearn OR lrskl, '
                             'logistic_regression OR lr, '
                             'naive_bayes_sklearn OR nbskl '
                             'naive_bayes OR nb).',
                        default='all')

    parser.add_argument('-tsize', '--train_size', action='store', dest='training_set_size',
                        help='Training set size (percentage). Should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the training split',
                        type=float,
                        default=0.8)

    parser.add_argument('-d', '--dataset', action='store', dest='dataset',
                        help='Database used '
                             '(Options: all, '
                             'ionosphere OR i '
                             'adult OR a '
                             'wine_quality OR wq'
                             'breast_cancer_diagnosis OR bcd).',
                        default='all')

    parser.add_argument('-plot_cost', '--plot_cost_vs_iterations', action='store_true', default=False,
                        dest='plot_cost_vs_iterations',
                        help='Plot different learning rates for gradient descent applied to logistic regression. '
                             'Use a threshold for change in the value of the cost function as termination criteria, '
                             'and plot the accuracy on train/validation set as a function of iterations of gradient '
                             'descent.')

    parser.add_argument('-lr', '--learning_rates_list', action='append', dest='learning_rates_list',
                        default=[], # if [] will use ['lr = .1', 'lr = .5', 'lr = 1']
                        help='Learning rates list used to plot cost versus iterations. For example: python main --classifier logistic_regression --dataset adult -plot_cost -lr 0.001 -lr 0.01 -lr 0.05 -lr 1',
                        type=float
                        )

    parser.add_argument('-heatmap', '--plot_heatmap', action='store_true', default=False,
                        dest='plot_heatmap',
                        help='Plot heatmaps for all datasets. Show the correlations between the datasets features (X). For example: python main.py --classifier naive_bayes --dataset wine_quality -heatmap')

    parser.add_argument('-save_logs', '--save_logs_in_file', action='store_true', default=False,
                        dest='save_logs_in_file',
                        help='Save logs in a file')

    parser.add_argument('-v', '--version', action='version', version='%(prog)s 1.0')

    options = parser.parse_args()

    print(parser.description, '\nRunning with options: ')
    print('\tClassifier =', options.classifier.upper())
    print('\tTraining set size =', options.training_set_size)
    print('\tDataset =', options.dataset.upper())
    print('\tSave logs in a file =', options.save_logs_in_file)
    print('\tPlot cost vs iterations =', options.plot_cost_vs_iterations)
    print('\tPlot heatmaps for all datasets (feature correlations) =', options.plot_heatmap)

    if options.plot_cost_vs_iterations:
        if options.learning_rates_list == []:
            print('\tLearning rates list =', [.1, .5, 1])
        else:
            print('\tLearning rates list =', options.learning_rates_list)

    if options.save_logs_in_file:
        if not os.path.exists('logs'):
            os.mkdir('logs')
        logging.basicConfig(filename='logs/all.log', format='%(asctime)s - %(levelname)s - %(message)s',
                            level=logging.INFO, datefmt='%m/%d/%Y %I:%M:%S %p')
    else:
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO,
                            datefmt='%m/%d/%Y %I:%M:%S %p')

    logging.info("Program started...")

    start = time.time()

    if options.classifier.upper() == Classifier.LOGISTIC_REGRESSION_SKLEARN.name or options.classifier.lower() == 'lrskl':
        run_classifier_given_dataset(Classifier.LOGISTIC_REGRESSION_SKLEARN, options.training_set_size)
    elif options.classifier.upper() == Classifier.LOGISTIC_REGRESSION.name or options.classifier.lower() == 'lr':
        run_classifier_given_dataset(Classifier.LOGISTIC_REGRESSION, options.training_set_size)
    elif options.classifier.upper() == Classifier.NAIVE_BAYES_SKLEARN.name or options.classifier.lower() == 'nbskl':
        run_classifier_given_dataset(Classifier.NAIVE_BAYES_SKLEARN, options.training_set_size)
    elif options.classifier.upper() == Classifier.NAIVE_BAYES.name or options.classifier.lower() == 'nb':
        run_classifier_given_dataset(Classifier.NAIVE_BAYES, options.training_set_size)
    else:
        print('\n\n\n==========================')
        print(Classifier.LOGISTIC_REGRESSION_SKLEARN.name)
        print('==========================')
        run_classifier(Classifier.LOGISTIC_REGRESSION_SKLEARN, Datasets.IONOSPHERE, options.training_set_size)
        run_classifier(Classifier.LOGISTIC_REGRESSION_SKLEARN, Datasets.ADULT, options.training_set_size)
        run_classifier(Classifier.LOGISTIC_REGRESSION_SKLEARN, Datasets.WINE_QUALITY, options.training_set_size)
        run_classifier(Classifier.LOGISTIC_REGRESSION_SKLEARN, Datasets.BREAST_CANCER_DIAGNOSIS, options.training_set_size)

        print('\n\n\n==========================')
        print(Classifier.LOGISTIC_REGRESSION.name)
        print('==========================')
        run_classifier(Classifier.LOGISTIC_REGRESSION, Datasets.IONOSPHERE, options.training_set_size)
        run_classifier(Classifier.LOGISTIC_REGRESSION, Datasets.ADULT, options.training_set_size)
        run_classifier(Classifier.LOGISTIC_REGRESSION, Datasets.WINE_QUALITY, options.training_set_size)
        run_classifier(Classifier.LOGISTIC_REGRESSION, Datasets.BREAST_CANCER_DIAGNOSIS, options.training_set_size)

        print('\n\n\n==========================')
        print(Classifier.NAIVE_BAYES_SKLEARN.name)
        print('==========================')
        run_classifier(Classifier.NAIVE_BAYES_SKLEARN, Datasets.IONOSPHERE, options.training_set_size)
        run_classifier(Classifier.NAIVE_BAYES_SKLEARN, Datasets.ADULT, options.training_set_size)
        run_classifier(Classifier.NAIVE_BAYES_SKLEARN, Datasets.WINE_QUALITY, options.training_set_size)
        run_classifier(Classifier.NAIVE_BAYES_SKLEARN, Datasets.BREAST_CANCER_DIAGNOSIS, options.training_set_size)

        print('\n\n\n==========================')
        print(Classifier.NAIVE_BAYES.name)
        print('==========================')
        run_classifier(Classifier.NAIVE_BAYES, Datasets.IONOSPHERE, options.training_set_size)
        run_classifier(Classifier.NAIVE_BAYES, Datasets.ADULT, options.training_set_size)
        run_classifier(Classifier.NAIVE_BAYES, Datasets.WINE_QUALITY, options.training_set_size)
        run_classifier(Classifier.NAIVE_BAYES, Datasets.BREAST_CANCER_DIAGNOSIS, options.training_set_size)

    if options.plot_cost_vs_iterations:
        cost_vs_iterations_plotting(options.learning_rates_list)

    if options.plot_heatmap:
        print('heatmap_plotting(plot_heatmap_values=True, load_dataset_with_extra_pre_processing=True)')
        heatmap_plotting(plot_heatmap_values=True, load_dataset_with_extra_pre_processing=True, save_plotting=False, plotting_path='plotting/plots/heatmaps/', save_csv_correlation_matrix=False)

        print('heatmap_plotting(plot_heatmap_values=True, load_dataset_with_extra_pre_processing=False)')
        heatmap_plotting(plot_heatmap_values=True, load_dataset_with_extra_pre_processing=False, save_plotting=False, plotting_path='plotting/plots/heatmaps/', save_csv_correlation_matrix=False)

        print('heatmap_plotting(plot_heatmap_values=False, load_dataset_with_extra_pre_processing=True)')
        heatmap_plotting(plot_heatmap_values=False, load_dataset_with_extra_pre_processing=True, save_plotting=False, plotting_path='plotting/plots/heatmaps/', save_csv_correlation_matrix=False)

        print('heatmap_plotting(plot_heatmap_values=False, load_dataset_with_extra_pre_processing=False)')
        heatmap_plotting(plot_heatmap_values=False, load_dataset_with_extra_pre_processing=False, save_plotting=False, plotting_path='plotting/plots/heatmaps/', save_csv_correlation_matrix=False)

    print('\n\nDONE!')

    msg = "Program finished. It took {} seconds".format(time.time() - start)
    if options.save_logs_in_file:
        logging.info(msg)
    else:
        print(msg)
