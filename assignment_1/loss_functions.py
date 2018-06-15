import numpy as np


def mean_squared_error(predicted_output, target_matrix):
    delta = predicted_output - target_matrix
    squared_errors = np.sum(np.square(delta), axis=0)
    return np.mean(squared_errors)
