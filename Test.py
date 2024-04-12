from datetime import datetime
from stratified_cv import StratifiedCV
from majority_voting_classifier import BaselineModel
from mean_model_classifier import MeanModel
from load_data_function import load_data
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

# Data processing:
bad_channels_strategy = "all"
short_channel_correction = True
negative_correlation_enhancement = True
threshold = 3
startTime = 7.5
stopTime = 12.5
K = 3

# Plotting and saving:
plot_epochs = False
plot_std_fNIRS_response = False
plot_accuracy_across_k_folds = True

save_plot_epochs = False
save_plot_std_fNIRS_response = False
save_plot_accuracy_across_k_folds = True
save_results = True

############################

def epoch_plot(epochs, epoch_type: str, bad_channels_strategy: str, save : bool, combine_strategy: str = "mean", threshold = None, data_set : str = "data_name"):

    """Plot epochs for one or multiple patients

    PARAMETERS
    ----------
    epochs : epoch element from mne pipeline
        epochs object to be plotted.
    epoch_type : str
        Type to be plotted, e.g. "Tapping" or "Noise". Dependant on the specific dataset.
    bad_channels_strategy : str
        Way to deal with differing bad channels according different epoch elements. Choose between "delete" or "all".
    combine_strategy : str
        Strategy for combining epochs, default is mean.
    save : bool.
        Whether to save the plot or not.
    """
    # Check if bad_channels_strategy is valid
    if bad_channels_strategy not in ("delete", "all", "threshold"):
        raise ValueError("Invalid bad_channels_strategy. Please use 'delete', 'all' og 'threshold'.")
    
    # Check if combine_strategy is valid
    if combine_strategy not in ("mean", "median", "sum", "gfp"):
        raise ValueError("Invalid combine_strategy. Please use 'mean', 'median', or 'sum'.")
    
    # Check if save is a boolean
    if not isinstance(save, bool):
        raise ValueError("Invalid value for save. Please use True or False.")
    
    # Function implementation:
    if bad_channels_strategy == "delete":
        for i in range(len(epochs)):
            epochs[i].info['bads'] = []
        epochs = mne.concatenate_epochs(epochs)
        
    elif bad_channels_strategy == "all":
        bad_channels = []
        for i in range(len(epochs)):
            bad_channels.extend(epochs[i].info['bads'])
        bad_channels = list(set(bad_channels))
        for i in range(len(epochs)):
                epochs[i].info['bads'] = bad_channels
        epochs = mne.concatenate_epochs(epochs)
    elif bad_channels_strategy == "threshold":
        if threshold == None:
            raise ValueError(f"When using bad_channels_strategy {bad_channels_strategy}, you must input a threshold value as an int.")
        else:
            bad_channels = []
            for i in range(len(epochs)):
                bad_channels.extend(epochs[i].info['bads'])

            # Count occurrences of each bad channel
            channel_counts = Counter(bad_channels)

            # Keep only channels that occur more than twice
            bad_channels = [channel for channel, count in channel_counts.items() if count > 2]

            # Update epochs with filtered bad channels
            for i in range(len(epochs)):
                epochs[i].info['bads'] = bad_channels
            epochs = mne.concatenate_epochs(epochs)
    
    # Plot the epochs
    plots = epochs[epoch_type].plot_image(
        combine=combine_strategy,
        vmin=-30,
        vmax=30,
        ts_args=dict(ylim=dict(hbo=[-15, 15], hbr=[-15, 15])),
    )

    # Save each plot if save is True
    plots_folder = "CNN_image_preprocesing"
    if save:
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        Plot_types = ["Oxyhemoglobin", "Deoxyhemoglobin"]
        for plot_type, plot in zip(Plot_types, plots):
            filename = os.path.join(plots_folder, f"{epoch_type}_epochs_plot_{plot_type}_{bad_channels_strategy}_{data_set}_{current_datetime}.pdf")
            plot.savefig(filename)
            print(f"Plot {plot_type} saved as {filename}")
            plt.close(plot)  # Close the figure after saving
            
all_epochs, data_name, all_data, freq, data_types = load_data(data_set = data_set, short_channel_correction = short_channel_correction, negative_correlation_enhancement = negative_correlation_enhancement)
epoch_plot(all_epochs, epoch_type=epoch_type, combine_strategy=combine_strategy, save=save_plot_epochs, bad_channels_strategy=bad_channels_strategy, threshold = threshold, data_set = data_name)