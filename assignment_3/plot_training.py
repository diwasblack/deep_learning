import json
import os

import matplotlib.pyplot as plt


def plot_training_data():
    filename = "cifar10_training.json"

    with open(filename, "r") as file:
        training_data = json.load(file)

    losses = training_data['loss']
    accuracies = training_data['acc']

    iterations = list(range(1, len(losses)+1))

    plt.plot(iterations, losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    output_path = os.path.join("cifar10_loss.png")
    plt.savefig(output_path)

    plt.clf()
    plt.plot(iterations, accuracies)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    output_path = os.path.join("cifar10_accuracy.png")
    plt.savefig(output_path)


if __name__ == "__main__":
    plot_training_data()

