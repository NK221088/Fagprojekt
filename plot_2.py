import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from datetime import datetime


file_path1 = "./final_results/E_test_output_20240619_003144.csv"  # Update with your actual file path
file_path2 = "./final_results/E_test_output_20240618_211115.csv"  # Update with your actual file path
file_path3 = "./final_results/E_test_output_20240618_233126.csv"  # Update with your actual file path

# Load the CSV files into DataFrames
df1 = pd.read_csv(file_path1)
df2 = pd.read_csv(file_path2)
df3 = pd.read_csv(file_path3)

# List of dataframes
dfs = [df1, df2, df3]

# Loop through the dataframes and remove rows named "Mean" and "PosNeg"
for i in range(len(dfs)):
    dfs[i] = dfs[i][~dfs[i]['Model'].isin(['Mean', 'PosNeg'])]
    
# Assign cleaned dataframes back to original variables
df1, df2, df3 = dfs

def parse_accuracy_dict(accuracy_dict_str):
    """Parse the accuracy dictionary string into an actual dictionary."""
    return eval(accuracy_dict_str)

# Adjust the plotting function to specifically show 0.96 confidence intervals

def plot_of_accuracy_across_k_folds_shaded(dfs, save_plot: bool = False):
    # Create a dictionary to store accuracies for each classifier
    classifier_accuracies = {}

    for df in dfs:
        for _, row in df.iterrows():
            classifier = row['Model']
            fold_accuracies = parse_accuracy_dict(row['E_test_Accuracy'])
            if classifier not in classifier_accuracies:
                classifier_accuracies[classifier] = {}
            for fold_number, (accuracy, conf_interval) in fold_accuracies.items():
                if fold_number not in classifier_accuracies[classifier]:
                    classifier_accuracies[classifier][fold_number] = []
                classifier_accuracies[classifier][fold_number].append((accuracy, conf_interval))

    # Calculate mean accuracies and confidence intervals for each fold
    for classifier in classifier_accuracies:
        for fold_number in classifier_accuracies[classifier]:
            accuracies = [x[0] for x in classifier_accuracies[classifier][fold_number]]
            conf_intervals = [x[1] for x in classifier_accuracies[classifier][fold_number]]
            mean_accuracy = np.mean(accuracies)
            std_dev = np.std(accuracies)
            # Assuming 0.96 confidence interval
            ci = 1.96 * (std_dev / np.sqrt(len(accuracies)))
            classifier_accuracies[classifier][fold_number] = (mean_accuracy, ci)

    # Sort the accuracies by fold number for each classifier
    for classifier in classifier_accuracies:
        classifier_accuracies[classifier] = sorted(classifier_accuracies[classifier].items())
    
    not_included = []
    
    # Plot the accuracies with shaded confidence intervals
    for classifier, accuracies in classifier_accuracies.items():
        x_values = [x[0] + 1 for x in accuracies]  # Fold numbers (starting from 1)
        y_values = [x[1][0] for x in accuracies]  # Mean accuracies
        y_errors = [x[1][1] for x in accuracies]  # Mean confidence intervals
        plt.plot(x_values, y_values, marker='o', label=classifier)  # Plot the line
        plt.fill_between(x_values, np.array(y_values) - np.array(y_errors), np.array(y_values) + np.array(y_errors), alpha=0.2)  # Add shaded confidence intervals

    plt.xlabel("Fold-number")
    plt.ylabel("Test accuracy")
    plt.title("Test accuracy across {} folds".format(len(classifier_accuracies[list(classifier_accuracies.keys())[0]])))
    plt.xticks(np.arange(1, len(classifier_accuracies[list(classifier_accuracies.keys())[0]]) + 1, 1))  # Ensuring only integers on the x-axis
    plt.ylim(0, 1)  # Setting the y-axis scale from 0 to 1
    plt.legend()

    if save_plot:
        plots_folder = "Plots"
        os.makedirs(plots_folder, exist_ok=True)  # Create the directory if it does not exist
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = os.path.join(plots_folder, f"accuracy_plot_{current_datetime}.pdf")
        plt.savefig(filename)
        print(f"Final plot saved as {filename}")

    plt.show()

# Call the function with the dataframes
plot_of_accuracy_across_k_folds_shaded([df1, df2, df3], save_plot=True)
