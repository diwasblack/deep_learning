import numpy as np

from activations import (
    sigmoid_function, sigmoid_function_derivative, hyperbolic_tangent,
    hyperbolic_tangent_derivative
)


class LayerBase():
    """
    Base class for the neural network layers
    """

    # TODO Add bias in each layer

    def __init__(self, input_units, output_units, use_softmax=False):
        self.input_units = input_units
        self.output_units = output_units

        # Randomly initialize weights in the range [-0.5, 0.5]
        self.weights = -0.5 + np.random.rand(self.output_units, self.input_units)

        self.activation_function = None
        self.activation_function_derivative = None

        # Store the activation value calcuated during the forward pass
        self.activation_values = None

    def forward_pass(self, input_matrix):
        """
        Layer method to compute the activation values during forward pass
        """
        # Compute the linear combination of weight and layer input
        linear_combination = np.dot(self.weights, input_matrix)

        # Compute the activation value
        self.activation_values = self.activation_function(linear_combination)

        return self.activation_values

    def backward_pass(self):
        pass


class TanhLayer(LayerBase):
    def __init__(self, *args):
        super().__init__(*args)
        # Assign hyperbolic_tangent function to be the activation function
        self.activation_function = hyperbolic_tangent
        self.activation_function_derivative = hyperbolic_tangent_derivative
