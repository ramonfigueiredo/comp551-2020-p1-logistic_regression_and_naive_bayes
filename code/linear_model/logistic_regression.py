import numpy as np


# TODO: Add regularization
class LogisticRegression:
    def __init__(self, lr=.01, eps=1e-2, max_iter=5000):
        self.lr = lr
        self.eps = eps
        self.max_iter = max_iter
        self.w = np.zeros(0)

    # Logistic function (sigmoid)
    def logistic(self, z):
        return 1 / (1 + np.exp(-z))

    # Calculates de gradient over the whole dataset (full-batch)
    def gradient(self, X, y):
        yh = self.logistic(np.dot(X, self.w))
        grad = np.dot(X.T, yh - y)
        return grad

    # Uses full-batch gradient descent to fit the model to the data provided
    def fit(self, X, y):
        N, D = X.shape
        self.w = np.zeros(D)
        g = np.inf
        i = 0
        while np.linalg.norm(g) > self.eps and i < self.max_iter:
            g = self.gradient(X, y)
            self.w = self.w - self.lr * g
            i += 1

        print("\n\nGradient descent finished.")
        print("Gradient norm: ", np.linalg.norm(g))
        print("Number of iterations: ", i)

    # Returns the model prediction over a set of data points
    def predict(self, x_test):
        yh = self.logistic(np.dot(x_test, self.w))
        return (yh > 0.5).astype(int)

    def cost(self, X, y):
        z = np.dot(X, self.w)
        J = np.mean(y * np.log1p(np.exp(-z)) + (1 - y) * np.log1p(np.exp(z)))
        return J

    def score(self, X, y_true):
        y_pred = self.predict(X)
        return np.average(y_true == y_pred)
