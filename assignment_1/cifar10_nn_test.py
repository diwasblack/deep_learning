import logging

from cifar10 import CIFAR10NN


def test_mnist():
    logging.basicConfig(
        format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)

    cifar10_nn = CIFAR10NN()

    # Load a pretrained network from a file
    cifar10_nn.load_pretrained_network()

    # Load the test data seperated during the training
    x_test, y_test = cifar10_nn.load_test_data()
    print("Accuracy: {}".format(cifar10_nn.evaluate_performance(x_test, y_test)))


if __name__ == "__main__":
    test_mnist()
