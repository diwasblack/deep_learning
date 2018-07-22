import pickle
import logging

from depth.models import Sequential
from depth.layers import Convolution2D, Flatten, DenseLayer, MaxPooling
from depth.optimizers import ADAM
from depth.regularizers import L2Regularizer
from depth.helpers import vector_to_label
from depth.metrics import categorical_accuracy


class CIFAR10CNN():

    def __init__(self):
        self.test_data_filename = "cifar10_test.pkl"
        self.weights_filename = "cifar10_cnn_weights.pkl"

        self.nn = None

    def train_network(self, x_train, y_train):
        # Create a logger for training
        logger = logging.getLogger("cifar10")
        logger.setLevel(logging.INFO)

        fh = logging.FileHandler("cifar10_training.log")
        logger.addHandler(fh)

        # Construct the network
        optimizer = ADAM(lr=0.01)

        regularizer = L2Regularizer(0.01)

        self.nn = Sequential()
        self.nn.add_layer(Convolution2D(
            5, (3, 3), input_shape=(3, 32, 32), regularizer=regularizer))
        self.nn.add_layer(MaxPooling(pool_size=(2, 2)))
        self.nn.add_layer(Convolution2D(10, (3, 3), regularizer=regularizer))
        self.nn.add_layer(MaxPooling(pool_size=(2, 2)))
        self.nn.add_layer(Flatten())
        self.nn.add_layer(DenseLayer(units=32, regularizer=regularizer))
        self.nn.add_layer(DenseLayer(units=10, activation="softmax"))
        self.nn.compile(loss="cross_entropy", error_threshold=0.01,
                        optimizer=optimizer)

        self.nn.train(
            x_train, y_train, logging_frequency=1, max_epochs=20,
            training_logger=logger, update_frequency=100,
            layers_filename=self.weights_filename, mini_batch_size=256)

    def store_test_data(self, x_test, y_test):
        logging.info("Storing test data")
        # Store the test data
        with open(self.test_data_filename, "wb") as file:
            pickle.dump((x_test, y_test), file)

    def load_test_data(self):
        with open(self.test_data_filename, "rb") as file:
            x_test, y_test = pickle.load(file)
            return x_test, y_test

    def load_pretrained_model(self):
        logging.info("Trying to load pretrained model")
        self.nn = Sequential()
        self.nn.load_layer_weights(self.weights_filename)

    def test_network(self, x_test, y_test):
        predicted_output = self.nn.predict(x_test)

        # Convert predicted output vector to labels
        predicted_labels = vector_to_label(predicted_output)
        return categorical_accuracy(predicted_labels, y_test)
