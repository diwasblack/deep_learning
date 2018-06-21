import logging

from cifar10 import CIFAR10NN

from cifar10_dataset import get_CIFAR10_data


def train_mnist():
    logging.basicConfig(
        format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)

    # Load preprocessed dataset for CIFAR10
    cifar10_dataset= get_CIFAR10_data('cifar-10-batches-py')

    x_train = cifar10_dataset['X_train']
    y_train = cifar10_dataset['y_train']

    x_test = cifar10_dataset['X_test']
    y_test = cifar10_dataset['y_test']

    # Flatten and transform x_train and x_test
    x_train = x_train.reshape(x_train.shape[0], -1).T
    x_test = x_test.reshape(x_test.shape[0], -1).T

    # Create NN object for CIFAR 10 dataset
    cifar10_nn = CIFAR10NN()

    # Train network for CIFAR 10 data
    cifar10_nn.construct_nn()
    cifar10_nn.store_test_data(x_test, y_test)
    cifar10_nn.train(x_train, y_train)
    accuracy = cifar10_nn.evaluate_performance(x_test, y_test)
    print("Accuracy :{}".format(accuracy))


if __name__ == "__main__":
    train_mnist()
