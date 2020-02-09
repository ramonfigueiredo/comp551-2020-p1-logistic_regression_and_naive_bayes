import matplotlib.pyplot as plt

import seaborn as sns
import plotly.express as px
import numpy as np

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

# Method gets a numpy array and index number draws the histogram for all rows in that column index
def histogramfeature(X, index):
    f = plt.figure()
    plt.title('Mitoses')
    plt.xlabel('Number')
    plt.ylabel('Number of Samples')

    plt.xticks([0, 1,2,3,4,5,6,7,8,9,10])
    plt.hist(X[:, index], alpha=1, facecolor='g')
    plt.show()
    f.savefig("Mitoses.pdf", bbox_inches='tight')


# Draws histogram for y
def histogramfory(y):
    f = plt.figure()
    # sns.distplot(y, kde=False, rug=True ,bins=10);

    plt.title('Benign and Malignant Cells')
    plt.ylabel('Number of Patients')
    # plt.bar(width)
    plt.xticks([0,1], ['Benign', 'Malignant' ])
    bins = [-0.5 ,  0.5 ,1.5]
    plt.hist(y,bins=bins, alpha=1, facecolor='g', rwidth=1)
    plt.legend(loc='upper left');

    plt.show()
    f.savefig("cancerNumber.pdf", bbox_inches='tight')


# Draws the scatterplot for two features in the index Xindex and yxindex
def scatterPlot(X, Xindex, yx, yxindex):
    f = plt.figure()
    ax1 = f.add_subplot(111)
    ax1.scatter(X[:, Xindex], yx[:, yxindex], c='r', marker='s', label='first')
    plt.legend(loc='upper left');
    plt.show()
    f.savefig("fo5.pdf", bbox_inches='tight')


# Draws scatterplot for two features by seperating them compared to y values being 0 or 1.
def scatterPlotwithY(X, Xindex, yx, yxindex, y):
    f = plt.figure()
    ax1 = f.add_subplot(111)
    # tempzero = np.zeros(shape=(len(y), 2))
    # tempone = np.zeros(shape=(len(y), 2))
    # for i in range(len(y)):
    #     if y[i] == 0:
    #         tempzero[i,0] = X[i, Xindex]
    #         tempzero[i,1] = yx[i, yxindex]
    #     else :
    #         tempone[i,0] = X[i, Xindex]
    #         tempone[i,1] = yx[i, yxindex]
    # return tempone,tempzero

    listpositivex1 = []
    listpositivex2 = []

    listnegative1 = []
    listnegative2 = []

    for i in range(len(y)):
        if y[i] == 1:
            listpositivex1.append(X[i, Xindex])
            listpositivex2.append(yx[i, yxindex])
        else:
            listnegative1.append(X[i, Xindex])
            listnegative2.append(yx[i, yxindex])
    # return listpositive, listnegative
    f = plt.figure()
    ax1 = f.add_subplot(111)
    ax1.scatter(listpositivex1, listpositivex2, c='r', marker='s', label='positive')
    ax1.scatter(listnegative1, listnegative2, c='b', marker='s', label='negative')

    plt.legend(loc='upper left');
    plt.show()
    f.savefig("fo6.pdf", bbox_inches='tight')

    #
    #
    # ax1.scatter(X[:, Xindex], yx[:, yxindex], y, c='r', marker='s', label='first')
    # # ax1.scatter(yx[:, yxindex], y, c='b', marker='s', label='second')
    #
    # plt.legend(loc='upper left');
    # plt.show()
    # f.savefig("fo5.pdf", bbox_inches='tight')


X, y = get_dataset(Datasets.BREAST_CANCER_DIAGNOSIS)
# histogramfeature(X.astype(float), 1)
# histogramfory(y.astype(int))

# histogramfory(y)
histogramfeature(X,8)

# scatterPlotwithY(X, 4, X, 5, y)
