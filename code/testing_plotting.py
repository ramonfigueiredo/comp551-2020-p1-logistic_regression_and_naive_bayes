import matplotlib.pyplot as plt

import seaborn as sns

# settings for seaborn plotting style
sns.set(color_codes=True)
# settings for seaborn plot sizes
sns.set(rc={'figure.figsize': (5, 5)})

from datasets.load_dataset import get_dataset
from datasets.load_dataset import Datasets


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

    # plt.xticks([0, 1,2,3,4,5,6,7,8,9,10])
    plt.hist(X[:, index], alpha=1, facecolor='g')
    plt.show()
    # f.savefig("Mitoses.pdf", bbox_inches='tight')


# Draws histogram for y
def histogramfory(y):
    f = plt.figure()
    # sns.distplot(y, kde=False, rug=True ,bins=10);

    plt.title('Wine')
    plt.ylabel('Number of Wine')
    # plt.bar(width)
    plt.xticks([0,1], ['Bad', 'Good' ])
    bins = [-0.5 ,  0.5 ,1.5]
    plt.hist(y,bins=bins, alpha=1, facecolor='g', rwidth=1)

    plt.show()
    # f.savefig("Wine.png", bbox_inches='tight')


# Draws the scatterplot for two features in the index Xindex and yxindex
def scatterPlot(X, Xindex, yx, yxindex):
    f = plt.figure()
    ax1 = f.add_subplot(111)
    ax1.scatter(X[:, Xindex], yx[:, yxindex], c='r', marker='s', label='first')
    plt.legend(loc='upper left');
    plt.show()
    # f.savefig("fo5.pdf", bbox_inches='tight')


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
    # f = plt.figure()
    ax1 = f.add_subplot(111)
    ax1.scatter(listpositivex1, listpositivex2, c='r', marker='s', label='Good')
    ax1.scatter(listnegative1, listnegative2, c='b', marker='s', label='Bad')

    plt.legend(loc='upper left');
    plt.show()
    # f.savefig("fo6.pdf", bbox_inches='tight')

    #
    #
    # ax1.scatter(X[:, Xindex], yx[:, yxindex], y, c='r', marker='s', label='first')
    # # ax1.scatter(yx[:, yxindex], y, c='b', marker='s', label='second')
    #
    # plt.legend(loc='upper left');
    # plt.show()
    # f.savefig("fo5.pdf", bbox_inches='tight')


X, y = get_dataset(Datasets.IONOSPHERE)
# histogramfeature(X.astype(float), 1)
# histogramfory(y.astype(int))
#
# X_train,X_test,y_train,y_test = load_adult(Datasets.ADULT)
# X= np.concatenate(X_train,X_test)

# histogramfory(y)
# histogramfeature(X,8)

listtemp = [0.5970, 0.5029,0.6311,0.6154, 0.5866,0.5459,
            0.6708, 0.5621,0.6185,0.6336 ,0.6735,0.6192, 0.6642,
            0.7483,0.5906,0.5180,0.5617,0.8256,0.5757,0.5705,
0.6882,0.5308,0.5994,
0.6699,
0.7412,0.5885,
0.6848,
0.6598,
0.5769,
0.5541,
0.6598,
0.5566,
0.5498,
0.5369,
0.6320,
0.5537,
0.5113,
0.5166,
0.5336,
0.5692,
0.6404,
0.5390,
0.5034,
0.5035,
0.6509,
0.5161,
0.5091,
0.5007,
0.5539,
0.5797,
0.6924,
0.5150]


# 6,4 0.5970
# 22,24 0.5029
# # 10,6 0.6311
# 14,6 0.6154
# 20,6 0.5866
# 32,6 0.5459
# 10,8 0.6708
# 12,8 0.5621
# 14,8 0.6185
# 16,8 0.6336
# 18,8 0.6735
# 12,10 0.6192
# 14,10 0.6642
# 16,10 0.7483
# 18,10 0.5906
# 20,10 0.5180
# 22,10 0.5617
# 14,12 0.8256
# 16,12 0.5757
# 18,12 0.5705
# 20,12 0.6882
# 22,12 0.5308
# 16,14 0.5994
# 18,14 0.6699
# 20,14 0.7412
# 22,14 0.5885
# 18,16 0.6848
# 20,16 0.6598
# 22,16 0.5769
# 24,16 0.5541
# 20,18 0.6598
# 22,18 0.5566
# 24,18 0.5498
# 21,19 0.5369
# 22,20 0.6320
# 24,20 0.5537
# 30,20 0.5113
# 32,20 0.5166
# 24,22 0.5336
# 26,22 0.5692
# 28,22 0.6404
# 30,22 0.5390
# 32,22 0.5034
# 26,24 0.5035
# 27,24 0.6509
# 30,24 0.5161
# 27,26 0.5091
# 31,27 0.5007
# 30,28 0.5539
# 32,28 0.5797
# 32,30 0.6924
# 33,31 0.5150




# def printMultiple()


# scatterPlotwithY(X, 2, X, 0, y)
# scatterPlotwithY(X, 2, X, 0, y)
# scatterPlot(X,2,X,0)
# histogramfeature(X,13 )
# classifier = LogisticRegression()
# classifier = GaussianNaiveBayes()
#
# for i in range(10):
#     histogramfeature(X,i)
# histogramfory(y)

#
# histogramfeature(X, 1)
# histogramfory(y)
# scatterPlot(X, 1, X, 3)
# scatterPlotwithY(X, 1, X, 3, y)

trainSize= [50,55,60,65,70,75,80,85,90,95]
ionosphereLR=[89.77,92.40,88.65,90.24,91.50,87.50,90.14,84.90,94.44,94.44]
ionosphereNB=[39.77,34.81,33.30,34.95,36.79,40.90,39.43,41.50,36.11,27.77]
adultLR=[43.40,43.21,43.82,43.39,43.96,43.97,44.04,43.44,44.55,42.13]
adultrNB=[26.06,27.63,26.00,25.61,25.48,25.21,24.91,25.98,25.31,25.55]
wineLR=[55.50,55.69,53.43,53.10,56.87,53.75,54.37,53.75,55.00,56.25]
wineNB=[93.87,93.88,94.53,95.00,93.33,95.25,93.43,93.33,90.62,93.75]
cancerLR=[98.24,98.05,97.81,96.25,94.63,96.49,98.54,96.11,98.55,97.15]
cancerNB=[95.90,96.10,98.17,96.66,96.09,95.90,98.54,98.05,95.65,97.14]
def printAccuracyComparison(X1,X2,size):
    f = plt.figure()
    plt.plot(size,X1,label='NB')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Train size (%)')
    plt.title('Cancer Accuracy Comparison')
    plt.plot(size,X2,color='r', label='LR')
    plt.legend(loc='upper left');
    plt.show()
    f.savefig("Cancer.pdf", bbox_inches='tight')
    f.savefig("Cancer.png", bbox_inches='tight')
printAccuracyComparison(cancerNB,cancerLR,trainSize)


