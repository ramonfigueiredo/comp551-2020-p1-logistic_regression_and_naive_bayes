import numpy as np


class BernoulliNaiveBayes:
    def __init__(self):
        self.log_prior = np.zeros(0)
        self.likelihood = np.zeros(0)

    # Fit the model
    def fit(self, X, y):
        N, C = y.shape
        D = X.shape[1]
        self.likelihood = np.zeros((C, 2)), np.zeros((D, 2))
        self.log_prior = np.log(np.mean(y, 0))[:, None]
        for n in range(N):
            for d in range(D):
                self.likelihood[d, y[n]] += 1

    # Returns the model prediction over a set of data points
    def predict(self, x_test):
        logp = np.log(self.prior) + np.sum(np.log(self.likelihood * x_test[:, None]), 0) + np.sum(
            np.log((1 - self.likelihood) * (1 - x_test[:, None])), 0)
        posterior = np.exp(logp)  # vector of size 2
        posterior /= np.sum(posterior)  # normalize
        return posterior  # posterior class probability


class GaussianNaiveBayes:
    def __init__(self):
        self.prior = np.zeros([2, 1])
        self.mu = np.zeros(0)
        self.s = np.zeros(0)

    # Fit the model
    def fit(self, X, y):
        N = y.shape
        D = X.shape[1]
        self.prior = np.zeros([2, 1])
        self.mu, self.s = np.zeros((2, D)), np.zeros((2, D))

        for c in range(2):
            self.mu[c, :] = np.mean(X[y == c], 0)
            self.s[c, :] = np.std(X[y == c], 0)

        self.s[:, :] += 1e-9 * np.var(X, axis=0).max()

        self.prior[1, 0] = np.mean(y)
        self.prior[0, 0] = 1 - self.prior[1]

    # Returns the model prediction over a set of data points
    def predict(self, x_test):
        log_likelihood = - np.sum(
            (np.log(self.s[:, None]) + .5 * (((x_test[None, :] - self.mu[:, None]) / self.s[:, None]) ** 2)),
        2)
        log_posterior = np.log(self.prior) + log_likelihood

        return (log_posterior[1] > log_posterior[0]).astype(int)
