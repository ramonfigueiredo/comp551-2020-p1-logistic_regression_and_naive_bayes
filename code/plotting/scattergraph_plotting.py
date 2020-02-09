import matplotlib.pyplot as plt


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
