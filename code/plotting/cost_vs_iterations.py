import time
import matplotlib.pyplot as plt
from datasets.load_dataset import get_dataset
from linear_model.logistic_regression import LogisticRegression
from model_selection.train_test_split import train_test_split
from preprocessing.standard_scaler import feature_scaling
from utils.datasets_enum import Datasets

if __name__ == '__main__':
    start = time.time()

    # Dataset list
    datasets = [Datasets.IONOSPHERE, Datasets.ADULT, Datasets.WINE_QUALITY, Datasets.BREAST_CANCER_DIAGNOSIS]

    # Initialize model
    classifier = LogisticRegression()

    for dataset_name in datasets:
        print("dataset: ", dataset_name)
        # Load the datasets
        X, y = get_dataset(dataset_name)

        # Split the datasets
        X_train, X_test, y_train, y_test = train_test_split(X, y, 0.8)

        # Feature Scalling
        X_train, X_test = feature_scaling(X_train, X_test)

        for lr in (.1, .5, 1):
        # for lr in (0.0001, 0.001, 0.01, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1):
            classifier.lr = lr
            # Fit the model to the dataset
            classifier.fit(X_train, y_train)
            # Plot the evolution of the cost during training
            plt.plot(range(len(classifier.cost_history)), classifier.cost_history)

        plt.legend(['lr = .1', 'lr = .5', 'lr = 1'], loc='upper right')
        # plt.legend(
        #     ['lr = .0001', 'lr = .001', 'lr = .01', 'lr = .1', 'lr = .2', 'lr = .3', 'lr = .4', 'lr = .5', 'lr = .6',
        #      'lr = .7', 'lr = .8', 'lr = .9', 'lr = 1'], loc='upper right')
        plt.title(dataset_name)
        plt.xlim((0, 100))
        plt.ylabel('Cost')
        plt.xlabel('Iterations')
        plt.show()

    print('\n\nDONE!')
    print('It took', time.time() - start, 'seconds.')
