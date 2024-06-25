import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from datetime import datetime

def parse_accuracy_dict(accuracy_dict_str):
    """Parse the accuracy dictionary string into an actual dictionary."""
    # Define a safe evaluation context
    context = {"array": np.array, "int64": np.int64}
    parsed_dict = eval(accuracy_dict_str, {"__builtins__": None}, context)
    return {k: (v[0], v[2]) for k, v in parsed_dict.items()}  # Adjusted to return only accuracy and confusion matrix

def calculate_confidence_interval(data, confidence=0.95):
    """Calculate the confidence interval for a list of data points."""
    mean = np.mean(data)
    std_err = np.std(data) / np.sqrt(len(data))
    margin_of_error = std_err * 1.96  # For 95% confidence
    return mean, margin_of_error

def plot_of_accuracy_across_k_folds_shaded(dfs, exclude_classifiers=None, save_plot: bool = False):
    if exclude_classifiers is None:
        exclude_classifiers = []

    # Create a dictionary to store accuracies for each classifier
    classifier_accuracies = {}

    for df in dfs:
        for _, row in df.iterrows():
            classifier = row['Model']
            if classifier in exclude_classifiers:
                continue
            fold_accuracies = parse_accuracy_dict(row['E_test_Accuracy'])
            if classifier not in classifier_accuracies:
                classifier_accuracies[classifier] = {}
            for fold_number, (accuracy, _) in fold_accuracies.items():
                if fold_number not in classifier_accuracies[classifier]:
                    classifier_accuracies[classifier][fold_number] = []
                classifier_accuracies[classifier][fold_number].append(accuracy)

    # Calculate mean accuracies and confidence intervals for each fold
    for classifier in classifier_accuracies:
        for fold_number in classifier_accuracies[classifier]:
            mean_accuracy, ci = calculate_confidence_interval(classifier_accuracies[classifier][fold_number])
            classifier_accuracies[classifier][fold_number] = (mean_accuracy, ci)

    # Sort the accuracies by fold number for each classifier
    for classifier in classifier_accuracies:
        classifier_accuracies[classifier] = sorted(classifier_accuracies[classifier].items())

    # Plot the accuracies with shaded confidence intervals
    for classifier, accuracies in classifier_accuracies.items():
        x_values = [x[0] + 1 for x in accuracies]  # Fold numbers (starting from 1)
        y_values = [x[1][0] for x in accuracies]  # Mean accuracies
        y_errors = [x[1][1] for x in accuracies]  # Confidence intervals
        plt.plot(x_values, y_values, marker='o', label=classifier)  # Plot the line
        plt.fill_between(x_values, np.array(y_values) - np.array(y_errors), np.array(y_values) + np.array(y_errors), alpha=0.2)  # Add shaded confidence intervals

    plt.xlabel("Participant")
    plt.ylabel("Test accuracy")
    plt.title("Test accuracy across {} participants".format(len(classifier_accuracies[list(classifier_accuracies.keys())[0]])))
    plt.xticks(np.arange(1, len(classifier_accuracies[list(classifier_accuracies.keys())[0]]) + 1, 1), rotation=0, fontsize=8)  # Ensuring only integers on the x-axis with rotation and smaller font size
    plt.ylim(0, 1)  # Setting the y-axis scale from 0 to 1
    plt.legend()

    if save_plot:
        plots_folder = "Final_results_for_report/Tapping"
        os.makedirs(plots_folder, exist_ok=True)  # Create the directory if it does not exist
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = os.path.join(plots_folder, f"accuracy_plot_{current_datetime}_Tapping.pdf")
        plt.savefig(filename)
        print(f"Final plot saved as {filename}")

    plt.show()

file_paths = [
    "./Final_results_for_report/Tapping/E_test_output_20240623_091633.csv",
    "./Final_results_for_report/Tapping/E_test_output_20240624_094528.csv",
    "./Final_results_for_report/Tapping/E_test_output_20240624_221420.csv"
]

df1 = pd.read_csv(file_paths[0])
df2 = pd.read_csv(file_paths[1])
df3 = pd.read_csv(file_paths[2])

# Call the function with the dataframe
plot_of_accuracy_across_k_folds_shaded([df1], exclude_classifiers=['PosNeg', "Mean"], save_plot=True)
