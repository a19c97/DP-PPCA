import numpy as np
import pandas as pd
from numpy.linalg import norm
import os

def load_data():
    data_path = './data/wine.data'
    data = pd.read_csv(data_path)
    data = data.to_numpy()
    data = data[:, 1:].T

    # Train-test split
    N = data.shape[1]
    train_N = int(np.floor(N*0.7))
    train_data = data[:, :train_N]
    test_data = data[:, train_N:]
    return train_data, test_data

def compute_cov_mtx(t, method=None):
    """
    Compute covariance matrix given d dimensional data points
    :param t: d-by-n data matrix, each column is a data point
    :param method: which private algorithm to use for computing
        covariance matrix; if None, do not add privacy
    :return: d-by-d covariance matrix
    """
    if method is None:
        # Standard, non-private way of computing cov
        return np.cov(t)
    # elif method == ''

class PPCA():
    def __init__(self, t, q):
        """

        :param t: d-by-n data matrix, each column is a data point
        :param q: dimension of latent variable,
            i.e.: number of dimensions to project data to
        """
        self.t = t
        self.q = q
        self.d = t.shape[0]
        self.N = t.shape[1]
        self.S = compute_cov_mtx(t)
        self.R = np.identity(q)

        # Find eigenvalues and eigenvectors
        eigs = np.linalg.eig(self.S)
        lambdas = eigs[0]
        U = eigs[1]

        # Sort eigenvalues and reorder eigenvectors
        idx = np.flip(np.argsort(lambdas))
        lambdas = lambdas[idx]
        U = U[:, idx]

        # Fit MLE estimates for sigma and W
        sigma_MLE = sum(lambdas[self.q:]) / (self.d - self.q)
        W_MLE = U[:, :self.q] @ \
                np.sqrt(np.diag(lambdas[:self.q]) - sigma_MLE * np.identity(self.q)) @ \
                self.R
        self.sigma = sigma_MLE
        self.W = W_MLE

        # Compute C and M matrices for future use
        self.C = self.W @ self.W.T + self.sigma * np.identity(self.d)

    def train_data_ll(self):
        return self.log_likelihood(self.t)

    def log_likelihood(self, t):
        """
        :param t: d-by-n matrix of observed (test) data
        :return: Log likelihood of observed data t under fitted PPCA model
        """
        S = np.cov(t)
        L = (-self.N/2) * (
            self.d * np.log(2*np.pi) + np.log(norm(self.C) + np.trace(self.C @ S))
        )
        return L

    def generate_sample(self):
        pass

if __name__ == '__main__':
    train_data, test_data = load_data()
    model = PPCA(train_data, 2)
    print(model.train_data_ll())
    print(model.log_likelihood(test_data))