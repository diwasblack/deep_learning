import pickle
import gzip
import logging

from depth.models import Sequential
from depth.layers import DenseLayer
from depth.helpers import one_hot_encoding, vector_to_label
from depth.metrics import categorical_accuracy
from depth.optimizers import SGD
from depth.regularizers import L2Regularizer


class CIFAR10NN():
    def __init__(self):
        self.nn = None
        self.input_data_dimension = 32 * 32 * 3
        self.output_data_dimension = 10

        self.learning_rate = 0.1
        self.error_threshold = 0.1
        self.momentum = 0.9
        self.regularization_coefficient = 0.01

        self.logging_frequency = 1
        self.update_frequency = 100

        self.layers_filename = "cifar10_nn_layers.pkl"
        self.test_data_filename = "cifar10_test_data.pkl"

        self.cifar10_logger = logging.getLogger("cifar10")
        self.cifar10_logger.setLevel(logging.INFO)

        fh = logging.FileHandler("cifar10_training.log")
        self.cifar10_logger.addHandler(fh)

    def construct_nn(self):
        # First construct an optimizer to use
        optimizer = SGD(lr=self.learning_rate, momentum=self.momentum)

        # Create L2 regularizer
        regularizer = L2Regularizer(self.regularization_coefficient)

        self.nn = Sequential()
        self.nn.add_layer(DenseLayer(
            units=128, activation="tanh",
            input_dimension=self.input_data_dimension, regularizer=regularizer
        ))
        self.nn.add_layer(DenseLayer(
            units=128, activation="relu",
            regularizer=regularizer))
        self.nn.add_layer(DenseLayer(
            units=128, activation="tanh",
            regularizer=regularizer))
        self.nn.add_layer(DenseLayer(
            units=self.output_data_dimension,
            activation="softmax", regularizer=regularizer))
        self.nn.compile(loss="cross_entropy",
                        error_threshold=self.error_threshold,
                        optimizer=optimizer)

    def train(self, train_data, train_labels):
        """
        Train the neural network constructed
        """

        # Convert target to one hot encoding vectors
        train_targets = one_hot_encoding(train_labels)

        self.nn.train(train_data, train_targets,
                      logging_frequency=self.logging_frequency,
                      update_frequency=self.update_frequency,
                      layers_filename=self.layers_filename,
                      training_logger=self.cifar10_logger)

    def store_test_data(self, x_test, y_test):
        """
        Store the test data in a file
        """

        # Store the test data on a file
        logging.info("Storing test data")

        with gzip.open(self.test_data_filename, "wb") as file:
            pickle.dump((x_test, y_test), file)

        logging.info("Test data saved")

    def load_test_data(self):
        """
        Load test data from pickle file
        """
        with gzip.open(self.test_data_filename, "rb") as file:
            x_test, y_test = pickle.load(file)

        return x_test, y_test

    def evaluate_performance(self, x_test, y_test):
        """
        Predict the output on the input data and evaluate accuracy
        """
        predicted_output = self.predict(x_test)
        target_output = y_test

        return categorical_accuracy(predicted_output, target_output)

    def load_pretrained_network(self):
        self.nn = Sequential()
        self.nn.load_layer_weights(self.layers_filename)

    def predict(self, input_matrix):
        predicted_output = self.nn.predict(input_matrix)

        labels = vector_to_label(predicted_output)

        return labels
