import logging

import numpy as np

from layers import TanhLayer, ReluLayer, LinearLayer, SigmoidLayer


class SequentialNeuralNet():
    """
    Implementation of sequential backpropagation neural network
    """

    def __init__(self, input_dimension):
        # List to hold the layers
        self.layers = []
        self.input_dimension = input_dimension
        self.learning_rate = None

    def add_layer(self, activation_function="tanh", units=64, use_softmax=False):
        if(not(self.layers)):
            previous_units = self.input_dimension
        else:
            previous_units = self.layers[-1].output_units

        if(activation_function == "tanh"):
            layer = TanhLayer(previous_units, units)

        elif(activation_function == "relu"):
            layer = ReluLayer(previous_units, units)

        elif(activation_function == "sigmoid"):
            layer = SigmoidLayer(previous_units, units)

        elif(activation_function == "linear"):
            layer = LinearLayer(previous_units, units)

        # Add layer to the list
        self.layers.append(layer)

    def compile(self, learning_rate=0.001, error_threshold=0.001):
        self.output_dimension = self.layers[-1].output_units
        self.learning_rate = learning_rate

        self.error_threshold = error_threshold

        self.number_of_layers = len(self.layers)

    def forward_pass(self, input_matrix):
        output = np.copy(input_matrix)
        for layer in self.layers:
            output = layer.forward_pass(output)

        return output

    def backpropagation(self, delta):
        """
        Propagate delta through the layers
        """
        for layer_index in range(self.number_of_layers-1, -1, -1):

            # Access layer object
            layer = self.layers[layer_index]

            # Propagate delta through layers
            delta = layer.backprop(delta, self.learning_rate)

    def train(self, input_matrix, target_matrix):
        number_of_iterations = 0
        while(True):
            # Propagate the input forward
            predicted_output = self.forward_pass(input_matrix)

            delta = predicted_output - target_matrix

            # Calculate the loss
            loss = np.linalg.norm(delta)

            if(loss < self.error_threshold):
                break

            if(number_of_iterations % 1000 == 0):
                logging.info("Cost: {}".format(loss))

            # Update weights using backpropagation
            self.backpropagation(delta)

            number_of_iterations += 1

    def predict(self, input_matrix):
        return self.forward_pass(input_matrix)
