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

def load_MNIST_data(get_labels=False):
    train_size = 10000
    test_size = 1000

    train_set = datasets.MNIST('./data', train=True, download=True).data.numpy()
    test_set = datasets.MNIST('./data', train=False, download=True).data.numpy()
    train_set = train_set.reshape(train_set.shape[0], 784).T
    test_set = test_set.reshape(test_set.shape[0], 784).T
    train_set = train_set[:, :train_size]
    test_set = test_set[:, :test_size]

    if get_labels:
        train_set_labels = datasets.MNIST('./data', train=True, download=True).targets.numpy()
        test_set_labels = datasets.MNIST('./data', train=False, download=True).targets.numpy()
        train_set_labels = train_set_labels[:train_size]
        test_set_labels = test_set_labels[:test_size]

    # Normalize data
    train_set = normalize_data(train_set)
    test_set = normalize_data(test_set)

    if get_labels:
        return train_set, test_set, train_set_labels, test_set_labels
    else:
        return train_set, test_set

def normalize_data(data):
    sample_max = np.max(data, axis=1).reshape(data.shape[0], 1)
    sample_min = np.min(data, axis=1).reshape(data.shape[0], 1)
    data = (data - sample_min) / (sample_max - sample_min + 1e-12)
    return data

if __name__ == '__main__':
    train_set, test_set, train_set_labels, test_set_labels = load_MNIST_data(get_labels=True)
    train_counts = {}
    test_counts = {}
    for label in range(10):
        train_counts[label] = 0
        test_counts[label] = 0
    for label in train_set_labels:
        train_counts[label] += 1
    for label in test_set_labels:
        test_counts[label] += 1
    print(train_counts)
    print(test_counts)