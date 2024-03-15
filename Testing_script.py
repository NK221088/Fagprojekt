from datetime import datetime
from stratified_cv import StratifiedCV
from majority_voting_classifier import BaselineModel
from mean_model_classifier import MeanModel
from fNirs_processesing_fNirs_motor import all_epochs, epochs, all_data, all_freq, data_name
# from fnirs_processing_AudioSpeechNoise import all_epochs, epochs, all_data, all_freq, data_name
from epoch_plot import epoch_plot
import mne

epoch_type = "Tapping"
combine_strategy = "mean"
save = True
bad_channels_strategy = "delete"
startTime = 9
K = 2
stopTime = 11
freq = all_freq
save_results = True

# Plot epochs and save results
epoch_plot(all_epochs, epoch_type=epoch_type, combine_strategy=combine_strategy, save=True, bad_channels_strategy=bad_channels_strategy)

# Run classifier and get results
results = StratifiedCV(all_data[epoch_type], all_data["Control"], startTime=startTime, K=K, stopTime=stopTime, freq=freq)

# Get current date and time
current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Construct filename with date and time
filename = f"{data_name}_{epoch_type}_results_{current_datetime}.txt"
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
        file.write("For the majority voting classifier: {}\n".format(round(results[1],2)))
        file.write("For the mean model classifier: {}\n".format(round(results[0],2)))

    print(f"Results saved as {filename}")