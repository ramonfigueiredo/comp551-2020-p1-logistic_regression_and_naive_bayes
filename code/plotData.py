import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from utils.datasets_enum import Datasets

from datasets.load_dataset import get_dataset, load_adult
import numpy as np
from scipy.stats import norm
import seaborn as sns

# settings for seaborn plotting style
sns.set(color_codes=True)
# settings for seaborn plot sizes
sns.set(rc={'figure.figsize': (5, 5)})

from datasets.load_dataset import get_dataset, Datasets

# df = pd.read_csv("datasets/data/breast-cancer-wisconsin/breast-cancer-wisconsin.data")


# X = df.iloc[:, 1].values
#
# y = df.iloc[:,2].values
#
#
# ax = sns.distplot(X)
# plt.show()


# ax.set(xlabel='Normal Distribution', ylabel='Frequency')
# For spline data
# xx = np.linspace(X.min(), X.max(), 1000)
# y_smooth = interp1d(X, y)(xx)
# y_smooth = interp1d(x, y, kind="cubic")(xx)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(xx, y_smooth, "r-")

# For Scatter Plot
# plt.scatter(X,y)


# Histogram
#
# f = plt.figure()
# plt.title('Clump Thickness')
# plt.xlabel('Size')
# plt.ylabel('Number of Samples')
# plt.hist(X, alpha=1, facecolor='g')
# plt.show()
# f.savefig("fo3.pdf", bbox_inches='tight')


# For counting the occurences for graph.
# (X =='g' ).sum()
# (y=='b').sum()

# Method gets
def histogramfeature(data, index):
    X = data.iloc[:, index].values
    f = plt.figure()
    plt.title('Thickness')
    plt.xlabel('Size')
    plt.hist(X, alpha=1, facecolor='g')
    plt.show()
    f.savefig("fo3.pdf", bbox_inches='tight')


def histogramfory(ydata):
    y = ydata.iloc[:].values
    f = plt.figure()
    plt.hist(y, alpha=1, facecolor='g')
    plt.show()
    f.savefig("fo4.pdf", bbox_inches='tight')


X, y = get_dataset(Datasets.BREAST_CANCER_DIAGNOSIS)
histogramfeature(X, 4)
# histogramfory(y)
