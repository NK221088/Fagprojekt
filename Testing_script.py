from datetime import datetime
from stratified_cv import StratifiedCV
from majority_voting_classifier import BaselineModel
from mean_model_classifier import MeanModel
from load_data_function import load_data
from epoch_plot import epoch_plot
from standard_fNIRS_response_plot import standard_fNIRS_response_plot
import mne
import os
from collections import Counter

data_set = "AudioSpeechNoise" #"fNirs_motor_full_data" #"fNIrs_motor", #"AudioSpeechNoise", 
epoch_type = "Speech"
combine_strategy = "mean"
bad_channels_strategy = "all"
threshold = 3
startTime = 7.5
stopTime = 12.5
K = 2
short_channel_correction = False
negative_correlation_enhancement = False
save_results = True
plot_epochs = True
plot_std_fNIRS_response = True
save_plot = True


all_epochs, data_name, all_data, freq, data_types = load_data(data_set = data_set, short_channel_correction = short_channel_correction, negative_correlation_enhancement = negative_correlation_enhancement)

# Plot epochs and save results
if plot_epochs:
    epoch_plot(all_epochs, epoch_type=epoch_type, combine_strategy=combine_strategy, save=save_plot, bad_channels_strategy=bad_channels_strategy, threshold = threshold, data_set = data_name)

# Plot the standard fNIRS response plot
if plot_std_fNIRS_response:
    standard_fNIRS_response_plot(all_epochs, data_types, combine_strategy=combine_strategy, save=save_plot, bad_channels_strategy=bad_channels_strategy, threshold = threshold, data_set = data_name)


results = StratifiedCV(all_data[epoch_type], all_data["Control"], startTime = startTime, K = K, stopTime = stopTime, freq = freq)

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
        file.write("For the majority voting classifier: {}\n".format(results[1],2))
        file.write("For the mean model classifier: {}\n".format(results[0],2))
        file.write("For the mean ps model classifier: {}\n".format(results[2],2))
        

    print(f"Results saved as {filename}")