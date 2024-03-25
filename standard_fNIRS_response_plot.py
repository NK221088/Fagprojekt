import mne
import matplotlib.pyplot as plt
import os
from collections import Counter
from datetime import datetime

def standard_fNIRS_response_plot(epochs, data_types: list, bad_channels_strategy: str, save: bool, combine_strategy: str = "mean", threshold=None, data_set: str = "data_name"):
    """Plot standard functional near-infrared spectroscopy (fNIRS) responses.

    This function plots the standard fNIRS responses for different conditions, such as tapping or control.

    Parameters:
    -----------
    epochs : Epoch object
        Epoch object containing the fNIRS data.
    data_types : list of str
        List of data types to be plotted, e.g., ["Tapping", "Control"].
    bad_channels_strategy : str
        Strategy for handling bad channels. Options: 'delete', 'all', or 'threshold'.
    save : bool
        Whether to save the plot.
    combine_strategy : str, optional
        Strategy for combining epochs. Default is 'mean'.
    threshold : int, optional
        Threshold value for the 'threshold' bad_channels_strategy.
    data_set : str, optional
        Name of the dataset. Default is 'data_name'.

    Raises:
    -------
    ValueError
        If bad_channels_strategy or combine_strategy is invalid.

    Notes:
    ------
    The evoked data for each condition (e.g., tapping/HbO, tapping/HbR, control/HbO, control/HbR)
    is computed and plotted.

    """
    # Check if bad_channels_strategy is valid
    if bad_channels_strategy not in ("delete", "all", "threshold"):
        raise ValueError("Invalid bad_channels_strategy. Please use 'delete', 'all' or 'threshold'.")
    
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
    
    # Create evoked data dictionary
    evoked_dict = {}
    for data_type in data_types:
        for hemoglobin in ("HbO", "HbR"):
            evoked_dict[f"{data_type}/{hemoglobin}"] = epochs[data_type].average(picks=hemoglobin.lower())
    
    # Rename channels until the encoding of frequency in ch_name is fixed
    for condition in evoked_dict:
        evoked_dict[condition].rename_channels(lambda x: x[:-4])

    color_dict = dict(HbO="#AA3377", HbR="b")
    styles_dict = dict(Control=dict(linestyle="dashed"))

    # Plot evoked data
    mne.viz.plot_compare_evokeds(
        evoked_dict, combine="mean", ci=0.95, colors=color_dict, styles=styles_dict
    )

    # Save the plot if specified
    if save:
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = os.path.join("Plots", f"standard_fNIRS_response_plot_{current_datetime}.pdf")
        plt.savefig(filename)
        print(f"Plot saved as {filename}")
        plt.close()  # Close the figure after saving