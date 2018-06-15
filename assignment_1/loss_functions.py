import numpy as np


def mean_squared_error(target_matrix, predicted_output):
    delta = predicted_output - target_matrix
    return np.linalg.norm(delta)
