import logging

from cifar10 import CIFAR10NN

from cifar10_dataset import load as load_dataset


def train_mnist():
    logging.basicConfig(
        format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)

    # Load dataset for CIFAR 10
    x_train, y_train, x_test, y_test = load_dataset('cifar-10-batches-py')

    # Flatten and transform x_train and x_test
    x_train = x_train.reshape(x_train.shape[0], -1).T
    x_test = x_test.reshape(x_test.shape[0], -1).T

    # Create NN object for CIFAR 10 dataset
    cifar10_nn = CIFAR10NN()

    # Train network for CIFAR 10 data
    cifar10_nn.construct_nn()
    accuracy = cifar10_nn.train_and_test(x_train, y_train, x_test, y_test)
    print("Accuracy :{}".format(accuracy))


if __name__ == "__main__":
    train_mnist()
