from datetime import datetime
from stratified_cv import StratifiedCV
from majority_voting_classifier import BaselineModel
from mean_model_classifier import MeanModel
from load_data_function import load_data
from epoch_plot import epoch_plot
from standard_fNIRS_response_plot import standard_fNIRS_response_plot
from Plotting_of_accuracy_across_k_folds import plot_of_accuracy_across_k_folds
import mne
import os
from collections import Counter
import numpy as np

############################
# Settings:
############################

# Data set:
data_set = "fNirs_motor_full_data"
epoch_type = "Tapping"
combine_strategy = "mean"
individuals = True

# Data processing:
bad_channels_strategy = "all"
short_channel_correction = True
negative_correlation_enhancement = True
threshold = 3
startTime = 7.5
stopTime = 12.5
K = 10

# Plotting and saving:
plot_epochs = False
plot_std_fNIRS_response = False
plot_accuracy_across_k_folds = False

save_plot_epochs = False
save_plot_std_fNIRS_response = False
save_plot_accuracy_across_k_folds = False
save_results = True

############################

if individuals:
    all_epochs, data_name, all_data, freq, data_types, all_individuals = load_data(data_set = data_set, short_channel_correction = short_channel_correction, negative_correlation_enhancement = negative_correlation_enhancement, individuals = individuals)
else:
    all_epochs, data_name, all_data, freq, data_types = load_data(data_set = data_set, short_channel_correction = short_channel_correction, negative_correlation_enhancement = negative_correlation_enhancement, individuals = individuals)
    
# Plot epochs and save results
if plot_epochs:
    epoch_plot(all_epochs, epoch_type=epoch_type, combine_strategy=combine_strategy, save=save_plot_epochs, bad_channels_strategy=bad_channels_strategy, threshold = threshold, data_set = data_name)

# Plot the standard fNIRS response plot
if plot_std_fNIRS_response:
    standard_fNIRS_response_plot(all_epochs, data_types, combine_strategy=combine_strategy, save=save_plot_std_fNIRS_response, bad_channels_strategy=bad_channels_strategy, threshold = threshold, data_set = data_name)

results = StratifiedCV(all_data[epoch_type], all_data["Control"], startTime = startTime, K = K, stopTime = stopTime, freq = freq)

if plot_accuracy_across_k_folds:
    plot_of_accuracy_across_k_folds(results_across_k_folds =  results, save_plot = save_plot_accuracy_across_k_folds)

results_string_format = {classifier: str(np.round(np.mean(result), 3)) + u"\u00B1" + str(np.round(1.96 * np.std(result)/np.sqrt(len(result)),3)) for (classifier, result) in results.items()}

# Get current date and time
current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Construct filename with date and time
results_folder = "Classifier_results" # Define the folder name
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
        file.write("K: {}\n".format(K))
        file.write("Stop Time: {}\n".format(stopTime))
        file.write("Frequency: {}\n".format(round(freq,3)))
        file.write("Results:\n")
        file.write("For the majority voting classifier: {}\n".format(results_string_format["MajorityVoting"],2))
        file.write("For the mean model classifier: {}\n".format(results_string_format["MeanModel"],2))
        file.write("For the mean ps model classifier: {}\n".format(results_string_format["PSModel"],2))
        file.write("For the mean SVM model classifier: {}\n".format(results_string_format["SVMModel"],2))
        file.write("For the ANN classifier: {}\n".format(results_string_format["ANNModel"],2))

    print(f"Results saved as {filename}")