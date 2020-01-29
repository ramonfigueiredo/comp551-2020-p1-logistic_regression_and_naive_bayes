import pandas as pd


def load_dataset(path, header='infer', sep=',', x_col_indices=slice(-1), y_col_indices=-1):
    dataset = pd.read_csv(path, header=header, sep=sep)
    X = dataset.iloc[:, x_col_indices].values
    y = dataset.iloc[:, y_col_indices].values
    return X, y


def load_ionosphere():
    return load_dataset('data/adult/adult.data', header=None)


def load_adult():
    return load_dataset('data/ionosphere/ionosphere.data', header=None)


def load_wine():
    return load_dataset('data/wine-quality/winequality-red.csv', sep=';')


def load_cancer():
    return load_dataset('data/breast-cancer-wisconsin/breast-cancer-wisconsin.data', header=None)

