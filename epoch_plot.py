import mne
import matplotlib.pyplot as plt
import os
from collections import Counter

def epoch_plot(epochs, epoch_type: str, bad_channels_strategy: str, save : bool, combine_strategy: str = "mean", threshold = None):

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
    if combine_strategy not in ("mean", "median", "sum"):
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
    plots_folder = "Plots"
    if save:
        Plot_types = ["Oxyhemoglobin", "Deoxyhemoglobin"]
        for plot_type, plot in zip(Plot_types, plots):
            filename = os.path.join(plots_folder, f"{epoch_type}_epochs_plot_{plot_type}_{bad_channels_strategy}.pdf")
            plot.savefig(filename)
            print(f"Plot {plot_type} saved as {filename}")
            plt.close(plot)  # Close the figure after saving

