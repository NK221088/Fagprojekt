import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

def plot_of_accuracy_across_k_folds(results_across_k_folds: dict, save_plot: bool = False):
    x_values = np.arange(1, len(results_across_k_folds['MeanModel']) + 1)  # assuming all models have same number of folds
    for (classifier, values) in results_across_k_folds.items():
        plt.plot(x_values, values, marker='o', label=classifier)  # adding dots at every data point
    plt.xlabel("Fold-number")
    plt.ylabel("Test accuracy")
    plt.title("Test accuracy across {} folds".format(len(results_across_k_folds['MeanModel'])))
    plt.xticks(np.arange(min(x_values), max(x_values)+1, 1))  # ensuring only integers on the x-axis
    plt.legend()

    if save_plot:
        plots_folder = "Plots"
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = os.path.join(plots_folder, f"accuracy_plot_{current_datetime}.pdf")
        plt.savefig(filename)
        print(f"Final plot saved as {filename}")
    plt.show()
