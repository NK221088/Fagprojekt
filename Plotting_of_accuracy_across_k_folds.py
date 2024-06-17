import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

def plot_of_accuracy_across_k_folds(results_across_k_folds: dict, save_plot: bool = False):
    # Create a dictionary to store accuracies for each classifier
    classifier_accuracies = {}

    for classifier, fold_accuracies in results_across_k_folds.items():
        for fold_number, (accuracy, _) in fold_accuracies.items():
            if classifier not in classifier_accuracies:
                classifier_accuracies[classifier] = []
            classifier_accuracies[classifier].append((fold_number, accuracy))

    # Sort the accuracies by fold number for each classifier
    for classifier in classifier_accuracies:
        classifier_accuracies[classifier].sort(key=lambda x: x[0])

    # Plot the accuracies
    for classifier, accuracies in classifier_accuracies.items():
        x_values = [x[0] + 1 for x in accuracies]  # Fold numbers (starting from 1)
        y_values = [x[1] for x in accuracies]  # Accuracies
        plt.plot(x_values, y_values, marker='o', label=classifier)  # Adding dots at every data point

    plt.xlabel("Fold-number")
    plt.ylabel("Test accuracy")
    plt.title("Test accuracy across {} folds".format(len(classifier_accuracies[list(classifier_accuracies.keys())[0]])))
    plt.xticks(np.arange(1, len(classifier_accuracies[list(classifier_accuracies.keys())[0]]) + 1, 1))  # Ensuring only integers on the x-axis
    plt.legend()

    if save_plot:
        plots_folder = "Plots"
        os.makedirs(plots_folder, exist_ok=True)  # Create the directory if it does not exist
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = os.path.join(plots_folder, f"accuracy_plot_{current_datetime}.pdf")
        plt.savefig(filename)
        print(f"Final plot saved as {filename}")

    plt.show()