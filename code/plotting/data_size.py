import time
import matplotlib.pyplot as plt
import numpy as np

from utils.datasets_enum import Datasets
from datasets.load_dataset import get_dataset
from model_selection.train_test_split import train_test_split
from preprocessing.standard_scaler import feature_scaling
from linear_model.logistic_regression import LogisticRegression
from linear_model.naive_bayes import GaussianNaiveBayes
from metrics.accuracy_score import evaluate_acc


def printAccuracyComparison(dataset_name, LR_acc, NB_acc, train_size):
    # Scale the values in the arrays
    LR_acc = np.multiply(LR_acc, 100)
    NB_acc = np.multiply(NB_acc, 100)
    train_size = np.multiply(train_size, 100)
    # Plot the arrays
    f = plt.figure()
    plt.plot(train_size, LR_acc, label='Logistic Regression')
    plt.plot(train_size, NB_acc, color='r', label='Naive Bayes')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Train size (%)')
    plt.ylim((20, 100))
    plt.title(dataset_name)
    plt.legend(loc='upper left')
    plt.show()
    # Save the figure
    path = "plotting/plots/data_size/" + str(dataset_name) + "_shuffle"
    f.savefig(path + ".pdf", bbox_inches='tight')
    f.savefig(path + ".png", bbox_inches='tight')


if __name__ == '__main__':
    start = time.time()

    datasets = [Datasets.IONOSPHERE, Datasets.ADULT, Datasets.WINE_QUALITY, Datasets.BREAST_CANCER_DIAGNOSIS]

    for ds in datasets:
        # Load and preprocess dataset
        X, y = get_dataset(ds)

        # Feature scaling
        X = feature_scaling(X)

        # Create the classifiers
        lr_classifier = LogisticRegression()
        nb_classifier = GaussianNaiveBayes()

        train_sizes = np.arange(0.05, 1, 0.05)
        lr_accuracy = []
        nb_accuracy = []

        for t in train_sizes:
            # Split into train and test
            X_train, X_test, y_train, y_test = train_test_split(X, y, t, shuffle=True)

            # Train and evaluate the models
            lr_classifier.fit(X_train, y_train)
            y_pred = lr_classifier.predict(X_test)
            lr_accuracy.append(evaluate_acc(y_test, y_pred))

            nb_classifier.fit(X_train, y_train)
            y_pred = nb_classifier.predict(X_test)
            nb_accuracy.append(evaluate_acc(y_test, y_pred))

        printAccuracyComparison(ds, lr_accuracy, nb_accuracy, train_sizes)

    print('\n\nDONE!')

    msg = "Program finished. It took {} seconds".format(time.time() - start)
