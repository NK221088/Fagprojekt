from datetime import datetime
from stratified_cv import StratifiedCV
from majority_voting_classifier import BaselineModel
from mean_model_classifier import MeanModel
from load_data_function import load_data
from epoch_plot import epoch_plot
from standard_fNIRS_response_plot import standard_fNIRS_response_plot
from Plotting_of_accuracy_across_k_folds import plot_of_accuracy_across_k_folds
from Individual_k_fold import individualKFold
import mne
import os
from collections import Counter
import numpy as np
from model import model
from seed import set_seeds
set_seeds()


############################
# Settings:
############################

# Data set:
data_set = "AudioSpeechNoise" #   "fNirs_motor_full_data" # "fNIRS_Alexandros_Healthy_data" # "fNIrs_motor" #      

epoch_type = "Speech"
combine_strategy = "mean"
individuals = False

# Data processing:
bad_channels_strategy = "all"
short_channel_correction = True
negative_correlation_enhancement = True
threshold = 3
startTime = 7.5
stopTime = 12.5
K = 5
interpolate_bad_channels = False

# Plotting and saving:
plot_epochs = False
plot_std_fNIRS_response = False
plot_accuracy_across_k_folds = True

save_plot_epochs = True
save_plot_std_fNIRS_response = True
save_plot_accuracy_across_k_folds = True
save_results = True


# Models
SVM = model(name = "SVM")
ANN = model(name = "ANN")
Mean = model(name = "Mean")
Baseline = model(name = "Baseline")
PosNeg = model(name = "PosNeg")
CNN = model(name = "CNN")

SVM.theta = {"kernel": ["rbf"], "C": [1.0], "gamma": ['scale'], "degree": [3], "coef0": [0.0]}
ANN.theta = {}
Mean.theta = {}
Baseline.theta = {}
PosNeg.theta = {}
CNN.theta = {"base_learning_rate": [0.01], "number_of_layers": [100], "batch_size": [32]}

modelList = [SVM, Baseline]

############################

if individuals:
    all_epochs, data_name, all_data, freq, data_types, all_individuals = load_data(data_set = data_set, short_channel_correction = short_channel_correction, negative_correlation_enhancement = negative_correlation_enhancement, individuals = individuals, interpolate_bad_channels=interpolate_bad_channels)
else:
    all_epochs, data_name, all_data, freq, data_types = load_data(data_set = data_set, short_channel_correction = short_channel_correction, negative_correlation_enhancement = negative_correlation_enhancement, individuals = individuals, interpolate_bad_channels=interpolate_bad_channels)
    
# Plot epochs and save results
if plot_epochs:
    epoch_plot(all_epochs, epoch_type=epoch_type, combine_strategy=combine_strategy, save=save_plot_epochs, bad_channels_strategy=bad_channels_strategy, threshold = threshold, data_set = data_name)

# Plot the standard fNIRS response plot
if plot_std_fNIRS_response:
    standard_fNIRS_response_plot(all_epochs, data_types, combine_strategy=combine_strategy, save=save_plot_std_fNIRS_response, bad_channels_strategy=bad_channels_strategy, threshold = threshold, data_set = data_name)

if individuals:
    results = individualKFold(individual_data = all_individuals, epoch_type=epoch_type, startTime = startTime, stopTime = stopTime)
else:
    results = StratifiedCV(modelList, all_data[epoch_type], all_data["Control"], startTime = startTime, K = K, stopTime = stopTime, freq = freq)

# Extract and format the results for each classifier
results_string_format = {}
for key, value in results.items():
    classifier = key[0]
    if classifier not in results_string_format:
        results_string_format[classifier] = []
    results_string_format[classifier].append(value[0])  # Append the accuracy

if plot_accuracy_across_k_folds:
    plot_of_accuracy_across_k_folds(results_across_k_folds =  results, save_plot = save_plot_accuracy_across_k_folds)



# Format the results with mean and confidence intervals
formatted_results = {classifier: f"{np.mean(acc):.3f} ± {1.96 * np.std(acc) / np.sqrt(len(acc)):.3f}" for classifier, acc in results_string_format.items()}

# Get current date and time
current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Construct filename with date and time
results_folder = "Classifier_results"  # Define the folder name
os.makedirs(results_folder, exist_ok=True)  # Create the directory if it does not exist
filename = os.path.join(results_folder, f"{data_name}_{epoch_type}_results_{current_datetime}.txt")
if save_results:
    with open(filename, "w") as file:
        file.write("Classifier Results:\n")
        file.write("====================\n")
        file.write(f"Data obtained at: {current_datetime}\n")
        file.write(f"Dataset: {data_name}\n")
        file.write(f"Epoch type used for classification: {epoch_type}\n")
        file.write(f"Combine Strategy: {combine_strategy}\n")
        file.write(f"Bad Channels Strategy: {bad_channels_strategy}\n")
        file.write(f"Start Time: {startTime}\n")
        file.write(f"K: {K}\n")
        file.write(f"Stop Time: {stopTime}\n")
        file.write(f"Frequency: {round(freq, 3)}\n")
        if individuals:
            file.write("The models were evaluated using hold one out with each patient.\n")
        file.write("Results:\n")
        for classifier, result in formatted_results.items():
            file.write(f"For the {classifier} classifier: {result}\n")

    print(f"Results saved as {filename}")
