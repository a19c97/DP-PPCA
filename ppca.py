import numpy as np
import pandas as pd
import torch
from numpy.linalg import norm
import os
import matplotlib.pyplot as plt
from load_data import load_MNIST_data, load_tabular_data
from cov_mtx import compute_cov_mtx
import pickle




class PPCA():
    def __init__(self, t, test_t, q, cov_mtx_method=None, epsilon=None, delta=None, m_bound=None):
        """

        :param t: d-by-n data matrix, each column is a data point
        :param q: dimension of latent variable,
            i.e.: number of dimensions to project data to
        """
        self.t = t
        self.test_t = test_t
        self.q = q
        self.d = t.shape[0]
        self.N = t.shape[1]
        self.S = compute_cov_mtx(
            t, method=cov_mtx_method, epsilon=epsilon, delta=delta, m_bound=m_bound
        )
        self.R = np.identity(q)
        self.mu = np.mean(self.t, axis=1).reshape(self.d, 1)

        # Find eigenvalues and eigenvectors
        eigs = np.linalg.eig(self.S)
        lambdas = eigs[0].real
        U = eigs[1].real

        # Sort eigenvalues and reorder eigenvectors
        idx = np.flip(np.argsort(lambdas))
        lambdas = lambdas[idx]
        U = U[:, idx]
        # U = np.identity(self.d) # For Karim's intellectual curiosity

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

    def test_data_ll(self):
        return self.log_likelihood(self.test_t)

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
        :param t: d-by-n data matrix to project and then reconstruct
        :return: reconstructed d-by-n data matrix
        """
        if len(t.shape) == 1:
            t = t.reshape(t.shape[0], 1)
        t_hat = self.W @ np.linalg.inv(self.W.T @ self.W) @ self.W.T @ (t - self.mu) + self.mu
        return t_hat.real

    def compute_avg_recon_error(self, t):
        t_hat = self.reconstruct_sample(t)
        recon_error = np.linalg.norm(t_hat - t)
        return recon_error / t.shape[1]

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


def save_model(model):
    with open(model.filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(model, outp, pickle.HIGHEST_PROTOCOL)


def plot_MNIST_digit(pixels):
    if pixels.shape != (28, 28):
        pixels = pixels.reshape(28, 28)
    plt.imshow(pixels, cmap='gray', interpolation='none')
    plt.show()


def recon(idx, model):
    t1 = test_data[:, idx]
    t1_hat = model.reconstruct_sample(t1)
    img_list = [t1, t1_hat]
    for idx in range(2):
        img = img_list[idx].reshape(28, 28)
        plt.subplot(2, 1, idx + 1)
        plt.tight_layout()
        plt.imshow(img, cmap='gray', interpolation='none')
    plt.show()


if __name__ == '__main__':
    try:
        # Iterative eigenvector sampling method


        # train_data, test_data = load_MNIST_data()
        # model = PPCA(train_data, test_data, 100, cov_mtx_method='rejection_sampling', epsilon=10, m_bound=1)
        # model = PPCA(train_data, test_data, 100, cov_mtx_method='laplace', epsilon=0.1)
        # model = PPCA(train_data, test_data, 100, cov_mtx_method='analyze_gauss', epsilon=10, delta=0.1)
        # model = PPCA(train_data, test_data, 100)
        # recon(0)


        # train_data, test_data = load_tabular_data('wine')

        # MNIST experiments
        # epsilon_list = [1e-2, 1e-1, 0.5, 1]
        # delta_list = [0.001, 0.01, 0.1, 0.5]
        # train_data, test_data = load_MNIST_data()
        # fig, axs = plt.subplots(
        #     1, 4, figsize=(12, 4), gridspec_kw={'wspace': 0.4}
        # )
        #
        # for delta in delta_list:
        #     analyze_gauss_errors = {}
        #
        #     orig_model = PPCA(train_data, test_data, 100)
        #     orig_errors = orig_model.compute_avg_recon_error(orig_model.test_t)
        #
        #     for epsilon in epsilon_list:
        #         gauss_model = PPCA(
        #             train_data, test_data, 100, cov_mtx_method='analyze_gauss', epsilon=epsilon, delta=delta
        #         )
        #         analyze_gauss_errors[epsilon] = gauss_model.compute_avg_recon_error(gauss_model.test_t)
        #
        #     # Process errors and plot
        #     df = pd.DataFrame(analyze_gauss_errors, index=['Analyze Gauss']).transpose()
        #     df['Non-private'] = orig_errors
        #     for col in list(df.columns):
        #         plt.plot(df[col], label=col)
        #     plt.legend()
        #     plt.show()





        # Generate samples
        # for idx in range(5):
        #     img = model.generate_sample().reshape(28, 28)
        #     plt.subplot(1, 5, idx+1)
        #     plt.tight_layout()
        #     plt.imshow(img, cmap='gray', interpolation='none')
        # plt.show()

        # Reconstruct samples
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

