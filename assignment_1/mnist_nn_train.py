import logging

import cloudpickle
import numpy as np

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

    # Calculate the mean from the training data
    x_train_mean = np.mean(x_train, axis=1, dtype=x_train.dtype).reshape(-1, 1)

    # Subtract mean from training and test data
    x_train -= x_train_mean
    x_test -= x_train_mean

    # Train network for mnist data
    mnist_nn.construct_nn()
    mnist_nn.store_test_data(x_test, y_test, x_train_mean)
    mnist_nn.train(x_train, y_train)
    accuracy = mnist_nn.evaluate_performance(x_test, y_test)
    print("Accuracy :{}".format(accuracy))


if __name__ == "__main__":
    train_mnist()
