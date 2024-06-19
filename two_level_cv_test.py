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
import csv
import pandas as pd
import shutil


############################
# Settings:
############################

# Data set:
data_set = "fNIRS_Alexandros_Healthy_data" # "fNirs_motor_full_data" #  "fNirs_motor_full_data" #"fNirs_motor_full_data"
epoch_type = "Imagery"
combine_strategy = "mean"
individuals = True # CANNOT BE CHANGED IN THIS SCRIPT
if not individuals:
    raise Warning("This script can't run when the individual parameter is set to False.")

# Data processing:
bad_channels_strategy = "mean"
short_channel_correction = True
negative_correlation_enhancement = True
threshold = 3
startTime = 0
stopTime = 15
K2 = 5
interpolate_bad_channels = False

# Plotting and saving:
save_results = True

# Models
SVM = model(name = "SVM")
ANN = model(name = "ANN")
# Mean = model(name = "Mean")
Baseline = model(name = "Baseline")
# PosNeg = model(name = "PosNeg")
CNN = model(name = "CNN")

# ANN.theta = {
#      "neuron1": [60, 128],
#      "neuron2": [100, 300],
#      "layers": [6, 8],
#      "learning_rate": ["decrease", "clr"],
#      "layer_type": ["dense", "conv1d", "lstm"],
#      "activation_function": ["relu", "elu"],
#      "dropout_rate": [0.3]
# }

SVM.theta = {"kernel": ["rbf", "poly"], "C": list(np.logspace(-2, 10, 13)), "gamma": list(np.logspace(-9, 3, 13)), "degree": [2], "coef0": [0]}
ANN.theta = {
    "model": [1,2],
   "neuron1": [60,150,200],
   "neuron2": [100],
   "layers": [6],
   "learning_rate": ["decrease"],
   "use_transfer_learning": [True],
   "use_svm": [True],
}
Baseline.theta = {}
CNN.theta = {"base_learning_rate": [0.1, 0.001], "number_of_layers": [50, 100], "batch_size": [32]}

# Mean.theta = {}
# PosNeg.theta = {}

ANN_AND_SVM = True

# mean = model('Mean')
modelList = [SVM, ANN, Baseline, CNN]


all_epochs, data_name, all_data, freq, data_types, all_individuals = load_data(data_set = data_set, short_channel_correction = short_channel_correction, negative_correlation_enhancement = negative_correlation_enhancement, individuals = individuals, interpolate_bad_channels=interpolate_bad_channels)
accuracy, E_genList, E_test = two_level_cross_validation(modelList = modelList, K2 = K2, startTime = startTime, stopTime = stopTime, freq=freq, dataset = all_individuals, data_types=data_types)

# Get current date and time
current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

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
        file.write("Theta parameters:\n")
        for model in modelList:
            file.write("{}: {}\n".format(model.name, model.theta))
        if ANN_AND_SVM == True:
            file.write("The architecture used the SVM instead of last layers.\n")
        file.write("Results:\n")
        for models, accuracy in accuracy.items():
            file.write("For the {} classifier: {}\n".format(models, np.round(accuracy,2)))
    print(f"Results saved as {filename}")

# Function to format parameters as strings
def format_parameter(param):
    return ", ".join([f"{k}={v}" for k, v in dict(param).items()])

# Function to flatten the evaluation list
def flatten_evallist(evallist):
    flattened_data = []
    for idx, evaluation in enumerate(evallist):
        for model_name, parameters in evaluation.items():
            for param, accuracy in parameters.items():
                flattened_data.append({
                    "Evaluation": idx + 1,
                    "Model": model_name,
                    "Parameter": format_parameter(param),
                    "Accuracy": accuracy
                })
    return flattened_data

# Flatten the evaluation list
flat_data = flatten_evallist(E_genList)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Define the filename with the timestamp
filename_gen = f"E_genList_output_{timestamp}.csv"

# Write the flattened data to a CSV file
with open(filename_gen, 'w', newline='') as csvfile:
    fieldnames = ["Evaluation", "Model", "Parameter", "Accuracy"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    for row in flat_data:
        writer.writerow(row)

# Create a new CSV file for E_test
filename_test = f"E_test_output_{timestamp}.csv"

# Write the E_test dictionary to a CSV file
with open(filename_test, 'w', newline='') as csvfile:
    fieldnames = ["Model", "E_test_Accuracy"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    for model_name, accuracy in E_test.items():
        writer.writerow({"Model": model_name, "E_test_Accuracy": accuracy})

print(f"E_test results saved as {filename_test}")