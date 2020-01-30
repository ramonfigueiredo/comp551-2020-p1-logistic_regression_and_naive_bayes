import os
from enum import Enum, unique
import pandas as pd


@unique
class Datasets(Enum):
    IONOSPHERE = 1
    ADULT = 2
    WINE_QUALITY = 3
    BREAST_CANCER_DIAGNOSIS = 4


def get_dataset(dataset):
    if dataset == Datasets.IONOSPHERE:
        return load_ionosphere()
    elif dataset == Datasets.ADULT:
        return load_adult()
    elif dataset == Datasets.WINE_QUALITY:
        return load_wine_quality()
    elif dataset == Datasets.BREAST_CANCER_DIAGNOSIS:
        return load_breast_cancer_diagnosis()
    else:
        raise Exception("Dataset does not exist")


def load_dataset(path, header='infer', sep=',', x_col_indices=slice(-1), y_col_indices=-1):
    dataset = pd.read_csv(path, header=header, sep=sep)
    # There are 16 instances in Groups 1 to 6 that contain a single missing (i.e., unavailable) attribute value,
    # denoted by "?". Replacing '?' values with the most frequent value (mode).
    dataset = dataset.replace('?', str(dataset[6].mode().values[0]))

    X = dataset.iloc[:, x_col_indices].values
    y = dataset.iloc[:, y_col_indices].values
    return X, y


def load_ionosphere():
    path = os.path.join( os.getcwd(), 'datasets/data/ionosphere/ionosphere.data')
    return load_dataset(path, header=None)


def load_adult():
    path = os.path.join(os.getcwd(), 'datasets/data/adult/adult.data')
    return load_dataset(path, header=None)


def load_wine_quality():
    path = os.path.join(os.getcwd(), 'datasets/data/wine-quality/winequality-red.csv')
    return load_dataset(path, sep=';')


def load_breast_cancer_diagnosis():
    path = os.path.join(os.getcwd(), 'datasets/data/breast-cancer-wisconsin/breast-cancer-wisconsin.data')
    return load_dataset(path, header=None)
