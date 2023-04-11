import numpy as np
import pandas as pd
import torch
from numpy.linalg import norm
import os
import matplotlib.pyplot as plt
from load_data import load_MNIST_data, load_tabular_data


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
            self.d * np.log(2*np.pi) + np.log(norm(self.C)) + np.trace(np.linalg.inv(self.C) @ S)
        )
        return L

    def generate_sample(self):
        pass

if __name__ == '__main__':
    # train_data, test_data = load_tabular_data('wine')

    try:
        train_data, test_data = load_MNIST_data()
        model = PPCA(train_data, 100)
        print(model.train_data_ll())
        print(model.log_likelihood(test_data))
    except:
        import sys, pdb, traceback
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

    # Plotting MNIST images
    # for i in range(6):
    #         idx = np.random.randint(low=0, high=len(test_data))
    #         plt.subplot(2, 3, i + 1)
    #         plt.tight_layout()
    #         pixels = test_data[idx].reshape(28, 28)
    #         plt.imshow(pixels, cmap='gray', interpolation='none')
    # plt.show()