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
        self.mu = np.mean(self.t, axis=1)

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
        self.sigma = sigma_MLE.real
        self.W = W_MLE.real

        # Compute C and M matrices for future use
        self.C = self.W @ self.W.T + self.sigma * np.identity(self.d)
        self.M = np.diag(np.diagonal(self.W.T @ self.W)) + self.sigma * np.identity(self.q)

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
        return L.real

    def reconstruct_sample(self, t):
        """
        Generate sample from fitted PPCA model
        :param t: d by n data matrix to project and then reconstruct
        :return: reconstructed d by n data matrix
        """
        t_hat = self.W @ np.linalg.inv(self.W.T @ self.W) @ self.W.T @ (t - self.mu) + self.mu
        return t_hat.real

    def generate_sample(self):
        """
        Performs ancestral sampling on fitted PPCA model
        :return: Generated sample
        """
        # Method 1: Generate x randomly
        x = np.random.multivariate_normal(np.zeros(self.q), np.identity(self.q))
        # Method 2: Generate x based on the x|t distribution, basically setting t to mu
        # Likely not as correct
        # M_inv = np.linalg.inv(self.M).real
        # x = np.random.multivariate_normal(np.zeros(self.q), self.sigma * M_inv)
        mean = self.W @ x + self.mu
        variance = self.sigma * np.identity(self.d)
        t = np.random.multivariate_normal(mean, variance).reshape(self.d, 1)
        return t

def plot_MNIST_digit(img_list, dims):

    if pixels.shape != (28, 28):
        pixels = pixels.reshape(28, 28)
    plt.imshow(pixels, cmap='gray', interpolation='none')
    plt.show()


if __name__ == '__main__':
    # train_data, test_data = load_tabular_data('wine')

    try:
        train_data, test_data = load_MNIST_data()
        model = PPCA(train_data, 100)
        for idx in range(5):
            img = model.generate_sample().reshape(28, 28)
            plt.subplot(1, 5, idx+1)
            plt.tight_layout()
            plt.imshow(img, cmap='gray', interpolation='none')
        plt.show()


        # t1 = test_data[:, 2]
        # t1_hat = model.reconstruct_sample(t1)
        # img_list = [t1, t1_hat]
        # for idx in range(2):
        #     img = img_list[idx].reshape(28, 28)
        #     plt.subplot(2, 1, idx+1)
        #     plt.tight_layout()
        #     plt.imshow(img, cmap='gray', interpolation='none')
        # plt.show()

        # plot_MNIST_digit(t1_hat)
        # print(model.train_data_ll())
        # print(model.log_likelihood(test_data))

        # Plotting MNIST images
        # for i in range(6):
        #     idx = np.random.randint(low=0, high=len(test_data[0]))
        #     plt.subplot(2, 3, i + 1)
        #     plt.tight_layout()
        #     pixels = test_data[:, idx].reshape(28, 28)
        #     plt.imshow(pixels, cmap='gray', interpolation='none')
        # plt.show()
    except:
        import sys, pdb, traceback
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

