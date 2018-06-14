import logging

import numpy as np

from layers import TanhLayer


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

    def backpropagation(self, input_matrix, predicted_output, target_matrix):
        eta = self.learning_rate

        # Assumes using the L2 distance for cost
        dloss_dy = predicted_output - target_matrix

        delta = np.dot(self.layers[-1].weights.T, dloss_dy)

        # Weight update for final layer
        weight_update = eta * \
            np.dot(dloss_dy, self.layers[-1].input_values.T)

        self.layers[-1].update_weights(weight_update)

        for layer_index in range(self.number_of_layers-2, -1, -1):

            # Access layer object
            layer = self.layers[layer_index]

            # Propagate delta through layers
            dloss_dz = layer.backward_pass(delta)

            # Calculate delta before weight update
            delta = np.dot(layer.weights.T, dloss_dz)

            # Update weight of the layer
            weight_update = eta * np.dot(dloss_dz, layer.input_values.T)
            layer.update_weights(weight_update)

    def train(self, input_matrix, target_matrix):
        number_of_iterations = 0
        while(True):
            # Propagate the input forward
            predicted_output = self.forward_pass(input_matrix)

            # Calculate the loss
            loss = np.linalg.norm(predicted_output-target_matrix)

            if(loss < self.error_threshold):
                break

            if(number_of_iterations % 1000 == 0):
                logging.info("Cost: {}".format(loss))

            # Update weights using backpropagation
            self.backpropagation(input_matrix, predicted_output, target_matrix)

            number_of_iterations += 1

    def predict(self, input_matrix):
        return self.forward_pass(input_matrix)
