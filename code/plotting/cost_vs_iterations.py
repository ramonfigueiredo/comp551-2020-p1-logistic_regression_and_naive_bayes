import time
import matplotlib.pyplot as plt
from datasets.load_dataset import get_dataset
from linear_model.logistic_regression import LogisticRegression
from model_selection.train_test_split import train_test_split
from preprocessing.standard_scaler import feature_scaling
from utils.datasets_enum import Datasets


def cost_vs_iterations_plotting(learning_rates_list):
    start = time.time()

    if not learning_rates_list:
        learning_rates_list = [.1, .5, 1]

    # Dataset list
    datasets = [Datasets.IONOSPHERE, Datasets.ADULT, Datasets.WINE_QUALITY, Datasets.BREAST_CANCER_DIAGNOSIS]

    # Initialize model
    classifier = LogisticRegression()

    for dataset_name in datasets:
        print("dataset: ", dataset_name)
        # Load the datasets
        X, y = get_dataset(dataset_name)

        # Feature Scalling
        X = feature_scaling(X)

        # Split the datasets
        X_train, X_test, y_train, y_test = train_test_split(X, y, 0.8, shuffle=True)

        for lr in learning_rates_list:
            classifier.lr = lr
            # Fit the model to the dataset
            classifier.fit(X_train, y_train)
            # Plot the evolution of the cost during training
            plt.plot(range(len(classifier.cost_history)), classifier.cost_history)

        legends = []
        for l in learning_rates_list:
            l_str = 'lr = ' + str(l)
            legends.append(l_str)

        plt.legend(legends, loc='upper right')
        plt.title(dataset_name)
        plt.xlim((0, 100))
        plt.ylabel('Cost')
        plt.xlabel('Iterations')
        plt.show()

    print('\n\nDONE!')
    print('It took', time.time() - start, 'seconds.')
