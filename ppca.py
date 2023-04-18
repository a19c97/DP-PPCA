import numpy as np
import pandas as pd
import torch
from numpy.linalg import norm
import os
import matplotlib.pyplot as plt
from load_data import load_MNIST_data, load_tabular_data
from cov_mtx import compute_cov_mtx
import pickle


LABEL_SIZE = 18
TITLE_SIZE = 32
SUB_TITLE_SIZE = 20
MARKER_SIZE = 20


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
        if cov_mtx_method is None:
            self.filename = 'non_private.pkl'
        elif cov_mtx_method == 'rejection_sampling':
            self.filename = f'IS_epsilon_{epsilon}.pkl'
        elif cov_mtx_method == 'analyze_gauss':
            self.filename = f'AG_epsilon_{epsilon}_delta_{delta}.pkl'
        elif cov_mtx_method == 'laplace':
            self.filename = f'LP_epsilon_{epsilon}.pkl'
        else:
            raise ValueError('Invalid cov_mtx_method!')

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
        x = np.random.multivariate_normal(np.zeros(self.q), np.identity(self.q)).reshape(self.q, 1)
        # Method 2: Generate x based on the x|t distribution, basically setting t to mu
        # Likely not as correct
        # M_inv = np.linalg.inv(self.M).real
        # x = np.random.multivariate_normal(np.zeros(self.q), self.sigma * M_inv)
        mean = self.W @ x + self.mu
        mean = mean.reshape(mean.shape[0], )
        variance = self.sigma * np.identity(self.d)
        t = np.random.multivariate_normal(mean, variance).reshape(self.d, 1)
        return t


def save_model(model):
    path = os.path.join('./saved_models', model.filename)
    with open(path, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(model, outp, pickle.HIGHEST_PROTOCOL)


def plot_MNIST_digit(pixels):
    if pixels.shape != (28, 28):
        pixels = pixels.reshape(28, 28)
    plt.imshow(pixels, cmap='gray', interpolation='none')
    plt.show()


def recon(idx, model):
    t1 = model.test_t[:, idx]
    t1_hat = model.reconstruct_sample(t1)
    img_list = [t1, t1_hat]
    for idx in range(2):
        img = img_list[idx].reshape(28, 28)
        plt.subplot(2, 1, idx + 1)
        plt.tight_layout()
        plt.imshow(img, cmap='gray', interpolation='none')
    plt.show()


def make_all_methods_graph():
    # MNIST experiments - main graph
    epsilon_list = [1e-3, 1e-2, 1e-1, 0.5, 1, 10]
    delta_list = [0.0001, 0.001, 0.01, 0.1]
    train_data, test_data = load_MNIST_data()
    orig_model = PPCA(train_data, test_data, 100)
    orig_errors = orig_model.compute_avg_recon_error(orig_model.test_t)

    fig, axs = plt.subplots(
        2, 2, figsize=(12, 4), gridspec_kw={'wspace': 0.5}
    )

    # Load models and build dataframes
    for idx, delta in enumerate(delta_list):
        analyze_gauss_errors = {}
        laplace_errors = {}
        is_errors = {}

        for epsilon in epsilon_list:
            # Load models from save files
            with open(f'./saved_models/AG_epsilon_{epsilon}_delta_{delta}.pkl', 'rb') as save_file:
                gauss_model = pickle.load(save_file)
            analyze_gauss_errors[epsilon] = gauss_model.compute_avg_recon_error(gauss_model.test_t)

            with open(f'./saved_models/LP_epsilon_{epsilon}.pkl', 'rb') as save_file:
                laplace_model = pickle.load(save_file)
            laplace_errors[epsilon] = laplace_model.compute_avg_recon_error(laplace_model.test_t)

            with open(f'./saved_models/IS_epsilon_{epsilon}.pkl', 'rb') as save_file:
                is_model = pickle.load(save_file)
            is_errors[epsilon] = is_model.compute_avg_recon_error(is_model.test_t)

        df1 = pd.DataFrame(analyze_gauss_errors, index=['AG']).transpose()
        df2 = pd.DataFrame(laplace_errors, index=['LP']).transpose()
        df3 = pd.DataFrame(is_errors, index=['IS']).transpose()
        df = pd.concat([df1, df2, df3], axis=1)
        df['Non-private'] = orig_errors
        print(f'delta: {delta}')
        print(df)
        df.to_csv(f'delta_{delta}_results.csv')

        # Find subplot indices
        row_idx = int(idx / 2)
        col_idx = int(idx % 2)

        for col in list(df.columns):
            axs[row_idx][col_idx].plot(
                df[col], label=col, marker='.', markersize=MARKER_SIZE
            )
            axs[row_idx][col_idx].legend(fontsize=LABEL_SIZE)

        axs[row_idx][col_idx].set_title(f'delta = {delta}', fontsize=SUB_TITLE_SIZE)
        axs[row_idx][col_idx].set_xlabel('Epsilon', fontsize=LABEL_SIZE)
        axs[row_idx][col_idx].set_ylabel('Recon Error', fontsize=LABEL_SIZE)
        axs[row_idx][col_idx].tick_params(axis='both', which='major', labelsize=LABEL_SIZE)

    fig.suptitle(
        'Average Test Set Reconstruction Error by Privacy Budget (All Methods)',
        fontsize=TITLE_SIZE
    )
    plt.show()


def make_ag_graph():
    # MNIST zoomed - in AG versus baseline
    epsilon_list = [0.0001, 0.0005, 0.001, 0.005, 0.0075, 0.01, 0.02, 0.03, 0.04, 0.05]
    delta_list = [0.0001, 0.001, 0.01, 0.1]
    train_data, test_data = load_MNIST_data()

    orig_model = PPCA(train_data, test_data, 100)
    orig_errors = orig_model.compute_avg_recon_error(orig_model.test_t)

    fig, axs = plt.subplots(
        2, 2, figsize=(12, 4), gridspec_kw={'wspace': 0.5}
    )

    for idx, delta in enumerate(delta_list):
        analyze_gauss_errors = {}

        for epsilon in epsilon_list:
            gauss_model = PPCA(
                train_data, test_data, 100, cov_mtx_method='analyze_gauss', epsilon=epsilon, delta=delta
            )
            analyze_gauss_errors[epsilon] = gauss_model.compute_avg_recon_error(gauss_model.test_t)

        # Process errors and plot
        df1 = pd.DataFrame(analyze_gauss_errors, index=['AG']).transpose()
        df = df1
        df['Non-private'] = orig_errors
        print(f'delta: {delta}')
        print(df)

        # Find subplot indices
        row_idx = int(idx / 2)
        col_idx = int(idx % 2)

        for col in list(df.columns):
            axs[row_idx][col_idx].plot(
                df[col], label=col, marker='.', markersize=MARKER_SIZE
            )
            axs[row_idx][col_idx].legend(fontsize=LABEL_SIZE)

        axs[row_idx][col_idx].set_title(f'delta = {delta}', fontsize=SUB_TITLE_SIZE)
        axs[row_idx][col_idx].set_xlabel('Epsilon', fontsize=LABEL_SIZE)
        axs[row_idx][col_idx].set_ylabel('Recon Error', fontsize=LABEL_SIZE)
        axs[row_idx][col_idx].tick_params(axis='both', which='major', labelsize=LABEL_SIZE)

    fig.suptitle(
        'Average Test Set Reconstruction Error by Privacy Budget (AG v. Baseline)',
        fontsize=TITLE_SIZE
    )
    plt.show()


def release_DP_data():
    epsilon_list = [0.1, 1, 10]
    delta = 0.0001
    train_data, test_data, train_labels, test_labels = load_MNIST_data(get_labels=True)

    for epsilon in epsilon_list:
        # Load models from save files
        with open(f'./saved_models/AG_epsilon_{epsilon}_delta_{delta}.pkl', 'rb') as save_file:
            gauss_model = pickle.load(save_file)

        with open(f'./saved_models/LP_epsilon_{epsilon}.pkl', 'rb') as save_file:
            laplace_model = pickle.load(save_file)

        with open(f'./saved_models/IS_epsilon_{epsilon}.pkl', 'rb') as save_file:
            is_model = pickle.load(save_file)

        gauss_output = gauss_model.reconstruct_sample(train_data)
        laplace_output = laplace_model.reconstruct_sample(train_data)
        is_output = is_model.reconstruct_sample(train_data)

        np.save(f'./DP_data/AG_epsilon_{epsilon}_delta_{delta}_data.npy', gauss_output)
        np.save(f'./DP_data/LP_epsilon_{epsilon}_data.npy', laplace_output)
        np.save(f'./DP_data/IS_epsilon_{epsilon}_data.npy', is_output)

        np.save('DP_data/train_labels.npy', train_labels)
        np.save('DP_data/train_data.npy', train_data)


def make_recon_sample_plots():
    epsilon = 1
    delta = 0.001
    # Load models from save files
    with open(f'./saved_models/AG_epsilon_{epsilon}_delta_{delta}.pkl', 'rb') as save_file:
        gauss_model = pickle.load(save_file)

    with open(f'./saved_models/LP_epsilon_{epsilon}.pkl', 'rb') as save_file:
        laplace_model = pickle.load(save_file)

    with open(f'./saved_models/IS_epsilon_{epsilon}.pkl', 'rb') as save_file:
        is_model = pickle.load(save_file)

    train_data, test_data = load_MNIST_data()
    original_model = PPCA(train_data, test_data, 100)
    models = [gauss_model, is_model, laplace_model, original_model]
    titles = [
        'Out-of-Sample Reconstruction from AG Method',
        'Out-of-Sample Reconstruction from IS Method',
        'Out-of-Sample Reconstruction from LP Method',
        'Out-of-Sample Reconstruction from Non-Private Baseline Method'
    ]

    indices = np.random.randint(low=0, high=test_data.shape[1], size=(1, 5))

    for model_idx, model in enumerate(models):
        fig, axs = plt.subplots(2, 5)
        fig.tight_layout()
        for i in range(5):
            t = model.test_t[:, i]
            t_hat = model.reconstruct_sample(t).reshape(28, 28)
            t = t.reshape(28, 28)
            axs[0][i].imshow(t, cmap='gray', interpolation='none')
            axs[1][i].imshow(t_hat, cmap='gray', interpolation='none')
        fig.suptitle(titles[model_idx], fontsize=TITLE_SIZE)
        plt.show()

if __name__ == '__main__':
    try:
        # release_DP_data()
        make_recon_sample_plots()
        # train_data, test_data = load_MNIST_data()
        # model = PPCA(train_data, test_data, 100, cov_mtx_method='rejection_sampling', epsilon=10, m_bound=1)
        # model = PPCA(train_data, test_data, 100, cov_mtx_method='laplace', epsilon=0.1)
        # model = PPCA(train_data, test_data, 100, cov_mtx_method='analyze_gauss', epsilon=10, delta=0.1)
        # model = PPCA(train_data, test_data, 100)
        # recon(0, model)

        # Generate samples
        # for idx in range(5):
        #     img = model.generate_sample().reshape(28, 28)
        #     plt.subplot(1, 5, idx + 1)
        #     plt.tight_layout()
        #     plt.imshow(img, cmap='gray', interpolation='none')
        # plt.show()

        # train_data, test_data = load_tabular_data('wine')



        # Run and save models for future use
        # for epsilon in epsilon_list:
        #     is_model = PPCA(
        #             train_data, test_data, 100, cov_mtx_method='rejection_sampling',
        #             epsilon=epsilon, m_bound=1
        #     )
        #     save_model(is_model)
        #
        #     laplace_model = PPCA(train_data, test_data, 100, cov_mtx_method='laplace', epsilon=epsilon)
        #     save_model(laplace_model)
        #
        #     for delta in delta_list:
        #         gauss_model = PPCA(
        #             train_data, test_data, 100, cov_mtx_method='analyze_gauss', epsilon=epsilon, delta=delta
        #         )
        #         save_model(gauss_model)




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

