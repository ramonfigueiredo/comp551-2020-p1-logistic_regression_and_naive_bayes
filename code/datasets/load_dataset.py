import pandas as pd


def load_dataset(path, header, x_col_indices, y_col_indices=-1):
    dataset = pd.read_csv(path, header=header)
    X = dataset.iloc[:, x_col_indices].values.to_numpy()
    y = dataset.iloc[:, y_col_indices].values.to_numpy()
    return X, y
