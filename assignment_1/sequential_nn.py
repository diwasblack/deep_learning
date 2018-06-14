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

    def add_layer(self, activation_function="tanh", units=64, use_softmax=False):
        if(not(self.layers)):
            previous_units = self.input_dimension
        else:
            previous_units = self.layers[-1].output_units

        if(activation_function == "tanh"):
            layer = TanhLayer(previous_units, units)

        # Add layer to the list
        self.layers.append(layer)

    def compile(self):
        self.output_dimension = self.layers[-1].output_dimension

    def predict(self, input_matrix):
        output = input_matrix
        for layer in self.layers:
            output = layer.forward_pass(output)

        return output


if __name__ == "__main__":
    input_data_dimension = 3
    number_of_samples = 10
    nn_object = SequentialNeuralNet(input_dimension=input_data_dimension)
    nn_object.add_layer(units=32)
    nn_object.add_layer(units=64)
    nn_object.add_layer(use_softmax=True)

    input_data = np.random.rand(input_data_dimension, 10)
    output = nn_object.predict(input_data)

    import pdb
    pdb.set_trace()
