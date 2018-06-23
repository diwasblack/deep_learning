import re
import sys
import os

import matplotlib.pyplot as plt

pattern = re.compile("Iteration:(\d*.\d*), Loss:(\d*.\d*), Accuracy:(\d*.\d*)")


def load_data(filepath):
    """
    Load data from the filepath specified
    """
    iterations = []
    losses = []
    accuracies = []
    with open(filepath, "r") as file:
        lines = file.readlines()

        for line in lines:
            result = re.search(pattern, line)

            iteration = int(result.group(1))
            loss = float(result.group(2))
            accuracy = float(result.group(3))

            iterations.append(iteration)
            losses.append(loss)
            accuracies.append(accuracy)

    return iterations, losses, accuracies


def generate_plots(filepath):

    head, filename_ext = os.path.split(filepath)
    filename, __ = filename_ext.split(".")

    iterations, losses, accuracies = load_data(filepath)

    plt.plot(iterations, losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")

    output_path = os.path.join(head, "{}_loss.png".format(filename))
    plt.savefig(output_path)

    plt.clf()
    plt.plot(iterations, accuracies)
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")

    output_path = os.path.join(head, "{}_accuracy.png".format(filename))
    plt.savefig(output_path)


if __name__ == "__main__":

    if(len(sys.argv) == 1):
        filepath = "mnist_training_5_minus_1.log"
    else:
        filepath = sys.argv[1]

    generate_plots(filepath)
