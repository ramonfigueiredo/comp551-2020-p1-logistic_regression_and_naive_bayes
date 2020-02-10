import matplotlib.pyplot as plt


# Method gets a numpy array and index number draws the histogram for all rows in that column index
def histogramfeature(X, index):
    f = plt.figure()
    plt.title('Thickness')
    plt.xlabel('Size')
    plt.hist(X[:, index], alpha=1, facecolor='g')
    plt.show()
    # f.savefig("fo3.pdf", bbox_inches='tight')


# Draws histogram for y
def histogramfory(y):
    f = plt.figure()
    plt.hist(y, alpha=1, facecolor='g')
    plt.show()
    # f.savefig("fo4.pdf", bbox_inches='tight')
