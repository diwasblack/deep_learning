import logging

from mnist import MNISTNN


def test_mnist():
    logging.basicConfig(
        format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)

    mnist_nn = MNISTNN()

    # Load a pretrained network from a file
    mnist_nn.load_pretrained_network()

    # Load the test data seperated during the training
    x_test, y_test, training_mean = mnist_nn.load_test_data()
    print("Accuracy: {}".format(mnist_nn.evaluate_performance(x_test, y_test)))


if __name__ == "__main__":
    test_mnist()
