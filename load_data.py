from torchvision import datasets
import pandas as pd
import numpy as np


def load_tabular_data(dataset_name):
    data_path = f'./data/{dataset_name}.data'
    data = pd.read_csv(data_path)
    data = data.to_numpy()
    data = data[:, 1:].T

    # Train-test split
    N = data.shape[1]
    train_N = int(np.floor(N*0.7))
    train_data = data[:, :train_N]
    test_data = data[:, train_N:]

    return train_data, test_data

def load_MNIST_data():
    train_set = datasets.MNIST('./data', train=True, download=True).data.numpy()
    test_set = datasets.MNIST('./data', train=False, download=True).data.numpy()
    train_set = train_set.reshape(train_set.shape[0], 784).T
    test_set = test_set.reshape(test_set.shape[0], 784).T
    train_set = train_set[:, :10000]
    test_set = test_set[:, :500]

    return train_set, test_set