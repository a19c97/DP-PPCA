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
    test_set = test_set[:, :1000]

    # Normalize data
    train_set = normalize_data(train_set)
    test_set = normalize_data(test_set)

    return train_set, test_set

def normalize_data(data):
    sample_max = np.max(data, axis=1).reshape(data.shape[0], 1)
    sample_min = np.min(data, axis=1).reshape(data.shape[0], 1)
    data = (data - sample_min) / (sample_max - sample_min + 1e-12)
    return data