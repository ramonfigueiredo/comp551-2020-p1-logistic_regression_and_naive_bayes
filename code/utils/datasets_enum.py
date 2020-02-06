from enum import Enum, unique


@unique
class Datasets(Enum):
    IONOSPHERE = 1
    ADULT = 2
    WINE_QUALITY = 3
    BREAST_CANCER_DIAGNOSIS = 4