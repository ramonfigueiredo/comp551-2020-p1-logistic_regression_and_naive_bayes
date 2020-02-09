from enum import Enum, unique


@unique
class Classifier(Enum):
    LOGISTIC_REGRESSION = 1
    NAIVE_BAYES = 2
    LOGISTIC_REGRESSION_SKLEARN = 3
    NAIVE_BAYES_SKLEARN = 4
