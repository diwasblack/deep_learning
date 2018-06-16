import logging

import cloudpickle

from mnist import MNISTNN
from depth.helpers import train_test_split


def train_mnist():
    logging.basicConfig(
        format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)

    # Load mnist dataset
    mnist_dataset = cloudpickle.load(open("mnist.data", "rb"))

    input_matrix = mnist_dataset.data.T
    output_matrix = mnist_dataset.target

    # Create NN object for MNIST dataset
    mnist_nn = MNISTNN()

    x_train, x_test, y_train, y_test = train_test_split(
        input_matrix, output_matrix)

    # Train network for mnist data
    mnist_nn.construct_nn()
    accuracy = mnist_nn.train_and_test(x_train, y_train, x_test, y_test)
    print("Accuracy :{}".format(accuracy))


if __name__ == "__main__":
    train_mnist()
