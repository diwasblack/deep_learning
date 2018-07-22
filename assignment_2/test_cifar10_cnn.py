import logging

from cifar10_cnn import CIFAR10CNN


def test_network():
    logging.basicConfig(
        format='%(asctime)s %(levelname)s %(message)s')

    cifar10_cnn = CIFAR10CNN()
    x_test, y_test = cifar10_cnn.load_test_data()

    # Load the pretrained model
    cifar10_cnn.load_pretrained_model()

    print(cifar10_cnn.test_network(x_test, y_test))


if __name__ == "__main__":
    test_network()
