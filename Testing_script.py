from datetime import datetime
from stratified_cv import StratifiedCV
from majority_voting_classifier import BaselineModel
from mean_model_classifier import MeanModel
from fNirs_processesing_fNirs_motor import data_fNirs_motor
from fnirs_processing_AudioSpeechNoise import all_epochs, epochs, all_data, all_freq, data_name
from fnirs_processing_AudioSpeechNoise_SCC import data_AudioSpeechNoise
from fnirs_processing_fnirs_motor_full_data import data_fNirs_motor_full_data
from epoch_plot import epoch_plot
import mne
import os
from collections import Counter

epoch_type = "Speech"
combine_strategy = "gfp"
save = False
bad_channels_strategy = "all"
threshold = 3
startTime = 7.5
K = 2
stopTime = 12.5
save_results = save
short_channel_correction = True
negative_correlation_enhancement = True
data_set = "AudioSpeechNoise"

def load_data(data_set : str, short_channel_correction : bool = None, negative_correlation_enhancement : bool = None):
    if data_set == "fNIrs_motor":
        all_epochs, data_name, all_data, all_freq = data_fNirs_motor(short_channel_correction, negative_correlation_enhancement)
        return all_epochs, data_name, all_data, all_freq
    if data_set == "AudioSpeechNoise":
        all_epochs, data_name, all_data, all_freq = data_AudioSpeechNoise(short_channel_correction, negative_correlation_enhancement)
        return all_epochs, data_name, all_data, all_freq
    if data_set ==  "fNirs_motor_full_data":
        all_epochs, data_name, all_data, all_freq = data_fNirs_motor_full_data(short_channel_correction, negative_correlation_enhancement)
        return all_epochs, data_name, all_data, all_freq
    

all_epochs, data_name, all_data, freq = load_data(data_set = data_set, short_channel_correction = short_channel_correction, negative_correlation_enhancement = negative_correlation_enhancement)

# Plot epochs and save results
epoch_plot(all_epochs, epoch_type=epoch_type, combine_strategy=combine_strategy, save=save, bad_channels_strategy=bad_channels_strategy, threshold = threshold, data_set = data_name)

results = StratifiedCV(all_data[epoch_type], all_data["Control"], startTime = startTime, K = K, stopTime = stopTime, freq = freq)

# Get current date and time
current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Construct filename with date and time
results_folder = "Classifier_results" # Define the folder name
filename = os.path.join(results_folder, f"{data_name}_{epoch_type}_results_{current_datetime}.txt")
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