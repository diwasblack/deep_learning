import logging

from depth.helpers import one_hot_encoding

from cifar10_dataset import get_CIFAR10_data
from cifar10_cnn import CIFAR10CNN


def train_network():
    logging.basicConfig(
        format='%(asctime)s %(levelname)s %(message)s')

    # Load preprocessed dataset for CIFAR10
    cifar10_dataset = get_CIFAR10_data('cifar-10-batches-py')

    x_train = cifar10_dataset['X_train']
    y_train = cifar10_dataset['y_train']

    x_test = cifar10_dataset['X_test']
    y_test = cifar10_dataset['y_test']

    cifar10_cnn = CIFAR10CNN()
    cifar10_cnn.store_test_data(x_test, y_test)

    # Convert labels to one hot encoding vectors
    y_targets = one_hot_encoding(y_train)

    cifar10_cnn.train_network(x_train, y_targets)


if __name__ == "__main__":
    train_network()
