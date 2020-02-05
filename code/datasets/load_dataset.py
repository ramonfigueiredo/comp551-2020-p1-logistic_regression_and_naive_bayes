import os
from enum import Enum, unique
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer


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


def load_dataset(path, header='infer', sep=',', remove_question_mark=False, x_col_indices=slice(-1), y_col_indices=-1):
    dataset = pd.read_csv(path, header=header, sep=sep)
    # dataset = genfromtxt(path, skip_header=header, delimiter=sep)

    if remove_question_mark:
        # Delete all rows from a dataframe containing question marks (?)
        # https://stackoverflow.com/questions/46269915/delete-all-rows-from-a-dataframe-containing-question-marks?rq=1
        dataset = dataset.replace({'?': np.nan}).dropna()

    X = dataset.iloc[:, x_col_indices].values
    y = dataset.iloc[:, y_col_indices].values
    return X, y


# -------------------------
#   IONOSPHERE DATASET
# -------------------------
# - Number of Instances: 351
# - Number of Attributes: 34 plus the class attribute
# - Attribute Information:
#    -- All 34 are continuous
#    -- The 35th attribute is either "good" or "bad"
# - Missing Values: None
#
def load_ionosphere():
    path = os.path.join(os.getcwd(), 'datasets/data/ionosphere/ionosphere.data')
    X, y = load_dataset(path, header=None)

    # Only the last column is categorical, with 2 categories.
    # Using label encoder to change it to 0 or 1
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    return X, y


# -------------------------
#   ADULT DATASET
# -------------------------
# - Number of Instances: 48842  (train=32561, test=16281)
# TODO: Duplicate or conflicting instances : 6
# - Number of Attributes: 14 plus the class attribute
# - Attribute Information:
#    -- Attributes 0, 2, 4, 10, 11, 12 are continuous
#    -- Attributes 1, 3, 5, 6, 7, 8, 13 are categorical
#    -- Attribute 9 (sex) is either "Male" or "Female"
#    -- Attribute 14 (output) is either ">50K" or "<=50K"
# - 3620 rows have missing values, that were replaced by '?'
#
def load_adult():
    path = os.path.join(os.getcwd(), 'datasets/data/adult/adult.data')
    X, y = load_dataset(path, header=None, remove_question_mark=True)

    # Apply label encoder to columns 9 and 14
    label_encoder = LabelEncoder()
    X[:, 9] = label_encoder.fit_transform(X[:, 9])
    y = label_encoder.fit_transform(y)

    # Apply one-hot encode to columns 1, 3, 5, 6, 7, 8, 13
    # The drop parameter makes it so the first category in each feature is dropped to avoid the dummy variable trap
    ct = ColumnTransformer(
        [('one_hot_encoder', OneHotEncoder(categories='auto', drop='first'), [1, 3, 5, 6, 7, 8, 13])],
        remainder='passthrough'
    )
    X = np.array(ct.fit_transform(X), dtype=np.float)

    return X, y


# -------------------------
#   WINE QUALITY DATASET
# -------------------------
# - Number of Instances: 1599
# - Number of Attributes: 11 plus the output attribute
# - Attribute Information:
#    -- All attributes are continuous
#    -- Output attribute is a score between 0 and 10 - must be converted to a binary class
# TODO: several of the attributes may be correlated, thus it makes sense to apply some sort of feature selection.
#
def load_wine_quality():
    path = os.path.join(os.getcwd(), 'datasets/data/wine-quality/winequality-red.csv')
    X, y = load_dataset(path, sep=';')

    y = (y >= 5).astype(int)

    return X, y


# -------------------------
#   BREAST CANCER DATASET
# -------------------------
# - Number of Instances: 699
# - Number of Attributes: 10 plus the class attribute
# - Attribute Information:
#    -- Attribute 0 is the sample code number, which is irrelevant
#    -- Attributes 1 to 9 are continuous from 1 to 10
#    -- Attribute 10 (class attribute) is 2 for benign, 4 for malignant
#  - There are 16 instances in Groups 1 to 6 that contain a single missing attribute value, denoted by "?".
#
def load_breast_cancer_diagnosis():
    path = os.path.join(os.getcwd(), 'datasets/data/breast-cancer-wisconsin/breast-cancer-wisconsin.data')

    # Replacing '?' values with the most frequent value (mode).
    X, y = load_dataset(path, header=None, remove_question_mark=True)

    return X, y