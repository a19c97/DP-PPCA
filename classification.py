import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


n_epochs = 100
patience = 5
batch_size_train = 100
batch_size_test = 2000
learning_rate = 0.01
momentum = 0.5
log_interval = 10


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


def run_classification(data_filename):
    # train_loader, test_loader = load_data()
    # train_size = len(train_loader.dataset)
    # test_size = len(test_loader.dataset)
    all_data = np.load(f'./DP_data/{data_filename}')
    all_labels = np.load(f'./DP_data/train_labels.npy')
    train_data, test_data, train_labels, test_labels = \
        train_test_split(all_data.T, all_labels.T, test_size=0.2, random_state=42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_size = train_data.shape[0]
    test_size = test_data.shape[0]

    network = Net()
    network.to(device)
    optimizer = optim.SGD(
        network.parameters(), lr=learning_rate, momentum=momentum
    )
    train_losses = []
    test_losses = []

    def train(epoch):
        network.train()
        total_batches = int(train_size/batch_size_train)

        for i in range(total_batches):
            data = train_data[i*batch_size_train:(i+1)*batch_size_train, :].astype(np.float32)
            data = data.reshape(batch_size_train, 1, 28, 28)
            target = train_labels[i*batch_size_train:(i+1)*batch_size_train]
            # Convert to torch tensors
            data = torch.from_numpy(data).to(device)
            target = torch.from_numpy(target).to(device)

            optimizer.zero_grad()
            output = network(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            if i % log_interval == 0:
                print(
                    'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, i * len(data), train_size,
                        100. * i / total_batches, loss.item())
                    )
                train_losses.append(loss.item())
                # torch.save(network.state_dict(), './MNIST_models/model.pth')
                # torch.save(optimizer.state_dict(), './MNIST_models/optimizer.pth')

    def test():
        network.eval()

        with torch.no_grad():
            data = torch.from_numpy(test_data.astype(np.float32)).to(device)
            data = data.reshape(batch_size_test, 1, 28, 28)
            target = torch.from_numpy(test_labels).to(device)
            output = network(data)
            preds = torch.max(output, 1).indices
            test_loss = F.nll_loss(output, target).item()
            correct_preds = preds.eq(target).sum()

        test_losses.append(test_loss)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
            test_loss, correct_preds, test_size,
            100. * correct_preds / test_size))
        return test_loss

    # Training
    counter = 0
    best_test_loss = np.inf
    for epoch in range(1, n_epochs + 1):
        train(epoch)
        test_loss = test()
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            counter = 0
            print(f"New best test_loss: {test_loss}!")
        else:
            counter += 1
            if counter >= patience:
                print(f"Patience exceeded, concluding training after epoch {epoch}")
                break
    return best_test_loss

if __name__ == '__main__':
    try:
        epsilon_list = [0.1, 1, 10]
        delta = 0.0001
        test_losses = {}
        test_losses['AG'] = {}
        test_losses['LP'] = {}
        test_losses['IS'] = {}
        for epsilon in epsilon_list:
            data_filename = f'AG_epsilon_{epsilon}_delta_{delta}_data.npy'
            test_losses['AG'][epsilon] = run_classification(data_filename)
            data_filename = f'LP_epsilon_{epsilon}_data.npy'
            test_losses['LP'][epsilon] = run_classification(data_filename)
            data_filename = f'IS_epsilon_{epsilon}_data.npy'
            test_losses['IS'][epsilon] = run_classification(data_filename)
        print(test_losses)
    except:
        import sys, pdb, traceback
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)