from load_data_function import load_data
from two_level_cross_validation import two_level_cross_validation
import sys
print(sys.version)
import mne
import os
from collections import Counter
from model import model
import numpy as np

from datetime import datetime

############################
# Settings:
############################


# Data set:
data_set = "fNirs_motor_full_data"
epoch_type = "Tapping"
combine_strategy = "mean"
individuals = True # CANNOT BE CHANGED IN THIS SCRIPT
if not individuals:
    raise Warning("This script can't run when the individual parameter is set to False.")

# Data processing:
bad_channels_strategy = "all"
short_channel_correction = True
negative_correlation_enhancement = True
threshold = 3
startTime = 7.5
stopTime = 12.5
K2 = 2

# Plotting and saving:
save_results = True

#Models
SVM = model(name = "SVM")
ANN = model(name = "ANN")


ANN.theta = [50]

modelList = [ANN, SVM]


all_epochs, data_name, all_data, freq, data_types, all_individuals = load_data(data_set = data_set, short_channel_correction = short_channel_correction, negative_correlation_enhancement = negative_correlation_enhancement, individuals = individuals)
accuracy, evallist, E_gen = two_level_cross_validation(modelList = modelList, K2 = K2, startTime = startTime, stopTime = stopTime, dataset = all_individuals)

# Get current date and time
current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def format_evallist(evallist):
    formatted_lines = []
    for idx, evaluation in enumerate(evallist):
        formatted_lines.append(f"Evaluation {idx + 1}:")
        for key, value in evaluation.items():
            model_name, fold_number, model_parameter = key
            score, count = value
            formatted_lines.append(f"  Model: {model_name}")
            formatted_lines.append(f"    Fold Number: {fold_number}")
            formatted_lines.append(f"    Model Parameter: {model_parameter}")
            formatted_lines.append(f"    Score: {score}")
            formatted_lines.append(f"    Count: {count}")
        formatted_lines.append("")  # Add a blank line for separation
    return "\n".join(formatted_lines)

# Construct filename with date and time
results_folder = "Two_level_classifier_results" # Define the folder name
filename = os.path.join(results_folder, f"{data_name}e_{epoch_type}_results_{current_datetime}.txt")
if save_results:
    with open(filename, "w") as file:
        file.write("Classifier Results:\n")
        file.write("====================\n")
        file.write("Data obtained at: {}\n".format(current_datetime))
        file.write("Dataset: {}\n".format(data_name))
        file.write("Epoch type used for classification: {}\n".format(epoch_type))
        file.write("Combine Strategy: {}\n".format(combine_strategy))
        file.write("Bad Channels Strategy: {}\n".format(bad_channels_strategy))
        file.write("Start Time: {}\n".format(startTime))
        file.write("K2: {}\n".format(K2))
        file.write("Stop Time: {}\n".format(stopTime))
        file.write("Frequency: {}\n".format(round(freq,3)))
        if individuals:
            file.write("The models were evaluated using hold one out with each patient.\n")
        file.write("Results:\n")
        for models, accuracy in accuracy.items():
            file.write("For the {} classifier: {}\n".format(models, np.round(accuracy,2)))
            
        file.write("E_gen/n")
        file.write(f'{E_gen}')
        
        file.write("Fold accuracies:\n")
        file.write(format_evallist(evallist))


    print(f"Results saved as {filename}")
