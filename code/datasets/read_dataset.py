import pandas as pd


adult_dataset = pd.read_csv('adult/adult.data', header = None)
print(adult_dataset)

X = adult_dataset.iloc[:, [2, 3]].values
y = adult_dataset.iloc[:, 4].values
print("X", X)
print("y", y)

breast_cancer_wisconsin_dataset = pd.read_csv('breast-cancer-wisconsin/breast-cancer-wisconsin.data', header = None)
print(breast_cancer_wisconsin_dataset)

X = adult_dataset.iloc[:, [2, 3]].values
y = adult_dataset.iloc[:, 4].values
print("X", X)
print("y", y)

ionosphere_dataset = pd.read_csv('ionosphere/ionosphere.data', header = None)
print(ionosphere_dataset)

X = adult_dataset.iloc[:, [2, 3]].values
y = adult_dataset.iloc[:, 4].values
print("X", X)
print("y", y)

wine_quality_dataset = pd.read_csv('wine-quality/winequality-red.csv')
print(ionosphere_dataset)

X = adult_dataset.iloc[:, [2, 3]].values
y = adult_dataset.iloc[:, 4].values
print("X", X)
print("y", y)