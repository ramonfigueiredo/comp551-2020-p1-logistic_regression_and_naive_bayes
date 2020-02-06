from enum import Enum, unique


@unique
class Classifier(Enum):
    LOGISTIC_REGRESSION = 1
    NAIVE_BAYES = 2
