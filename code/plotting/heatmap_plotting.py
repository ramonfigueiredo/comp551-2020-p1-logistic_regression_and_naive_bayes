import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from datasets.load_dataset import load_ionosphere, load_wine_quality, load_breast_cancer_diagnosis, open_adult_training_data, open_adult_test_data, load_dataset
from utils.datasets_enum import Datasets


def heatmap_plotting(print_correlation_matrix=True, plot_heatmap_values=False, show_plotting=True, save_plotting=False, plotting_path='heatmap.png', load_dataset_with_removed_columns=True):
    # Dataset list
    datasets = [Datasets.IONOSPHERE, Datasets.ADULT, Datasets.WINE_QUALITY, Datasets.BREAST_CANCER_DIAGNOSIS]

    for dataset_name in datasets:
        if dataset_name == Datasets.IONOSPHERE:
            path = os.path.join(os.getcwd(), 'datasets/data/ionosphere/ionosphere.data')
            if load_dataset_with_removed_columns:
                X_np, y_np = load_ionosphere()
            else: # load all dataset columns
                X_np, y_np = load_dataset(path, header=None)
            X = pd.DataFrame(data=X_np)

        if dataset_name == Datasets.ADULT:
            continue
            X_train = open_adult_training_data()
            X_test = open_adult_test_data()

            X = pd.concat([X_train, X_test])

            # X = np.concatenate((X_train, X_test), axis=0)
            # y = np.concatenate((y_train, y_test), axis=0)
            #
            # X, y = preprocess_adult_dataset(X, y)

        if dataset_name == Datasets.WINE_QUALITY:
            X_np, y_np = load_wine_quality()

            X = pd.DataFrame(data=X_np)

        if dataset_name == Datasets.BREAST_CANCER_DIAGNOSIS:
            continue
            path = os.path.join(os.getcwd(), 'datasets/data/breast-cancer-wisconsin/breast-cancer-wisconsin.data')

            if load_dataset_with_removed_columns:
                X_np, y_np = load_breast_cancer_diagnosis()
            else: # load all dataset columns
                X_np, y_np = load_dataset(path, header=None, remove_question_mark=True)
            X = pd.DataFrame(data=X_np)

        sns.set(style="white")

        # Compute the correlation matrix
        corr = X.corr()
        if print_correlation_matrix:
            print('\nCorrelation matrix:\n', corr)

        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(220, 10, as_cmap=True)

        heatmap_plot = sns.heatmap(corr, xticklabels=True, yticklabels=True, annot=plot_heatmap_values, cmap=cmap,
                                   vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})

        if show_plotting:
            plt.show()
        if save_plotting:
            fig = heatmap_plot.get_figure()
            fig.savefig(plotting_path)
