import numpy as np


class LogisticRegression:
    def __init__(self, X, y, lr=.01, eps=1e-2):
        self.X = X
        self.y = y
        self.lr = lr
        self.eps = eps

        self.N, self.D = X.shape
        self.w = np.zeros(self.D)

    def fit(self):
        self.gradient_descent()

    def predict(self, x_test):
        return self.logistic(np.dot(x_test, self.w))

    def cost(self):
        z = np.dot(self.X, self.w)
        J = np.mean(self.y * np.log1p(np.exp(-z)) + (1 - self.y) * np.log1p(np.exp(z)))
        return J

    def logistic(self, z):
        return 1 / (1 + np.exp(-z))

    def gradient(self):
        yh = self.logistic(np.dot(self.X, self.w))
        grad = np.dot(self.X.T, yh - self.y)
        return grad

    def gradient_descent(self):
        g = np.inf
        while np.linalg.norm(g) > self.eps:
            g = self.gradient()
            self.w = self.w - self.lr * g
