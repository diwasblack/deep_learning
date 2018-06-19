import gzip
import pickle
import logging

from depth.sequential import NeuralNet
from depth.helpers import one_hot_encoding, vector_to_label
from depth.metrics import categorical_accuracy
from depth.optimizers import SGD


class MNISTNN():
    def __init__(self):
        self.nn = None
        self.input_data_dimension = 784
        self.output_data_dimension = 10

        self.learning_rate = 1e-5
        self.error_threshold = 0.1
        self.learning_rate_decay = 0.9

        self.logging_frequency = 10
        self.update_frequency = 100

        self.layers_filename = "mnist_nn_layers.pkl"
        self.test_data_filename = "mnist_test_data.pkl"

        self.mnist_logger = logging.getLogger("mnist")
        self.mnist_logger.setLevel(logging.INFO)

        fh = logging.FileHandler("mnist_training.log")
        self.mnist_logger.addHandler(fh)

    def construct_nn(self):
        # First construct an optimizer to use
        optimizer = SGD(lr=self.learning_rate, decay=self.learning_rate_decay)

        self.nn = NeuralNet()
        self.nn.add_layer(units=32, activation_function="tanh",
                          input_dimension=self.input_data_dimension)
        self.nn.add_layer(units=32, activation_function="relu")
        self.nn.add_layer(units=32, activation_function="tanh")
        self.nn.add_layer(units=self.output_data_dimension,
                          activation_function="softmax")
        self.nn.compile(loss="cross_entropy",
                        error_threshold=self.error_threshold,
                        optimizer=optimizer)

    def train(self, train_data, train_labels):
        # Convert target to one hot encoding vectors
        train_targets = one_hot_encoding(train_labels)

        self.nn.train(train_data, train_targets,
                      logging_frequency=self.logging_frequency,
                      update_frequency=self.update_frequency,
                      layers_filename=self.layers_filename,
                      training_logger=self.mnist_logger)

    def train_and_test(self, x_train, y_train, x_test, y_test):
        """
        Class method to train and test the neural network for MNIST dataset
        """
        # Store the test data on a file
        logging.info("Storing test data")

        with gzip.open(self.test_data_filename, "wb") as file:
            pickle.dump((x_test, y_test), file)

        logging.info("Test data saved")

        # Train the neural network
        self.train(x_train, y_train)

        return self.evaluate_performance(x_test, y_test)

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
        self.nn = NeuralNet()
        self.nn.load_layer_weights(self.layers_filename)

    def predict(self, input_matrix):
        predicted_output = self.nn.predict(input_matrix)

        labels = vector_to_label(predicted_output)

        return labels
