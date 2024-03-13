import mne
import matplotlib.pyplot as plt

def epoch_plot(epochs, epoch_type: str, bad_channels_strategy: str, save : bool, combine_strategy: str = "mean"):

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
    if bad_channels_strategy not in ("delete", "all"):
        raise ValueError("Invalid bad_channels_strategy. Please use 'delete' or 'all'.")
    
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
    if bad_channels_strategy == "all":
        bad_channels = []
        for i in range(len(epochs)):
            bad_channels.extend(epochs[i].info['bads'])
        bad_channels = list(set(bad_channels))
        for i in range(len(epochs)):
                epochs[i].info['bads'] = bad_channels
        epochs = mne.concatenate_epochs(epochs)
    epochs[epoch_type].plot_image(
    combine=combine_strategy,
    vmin=-30,
    vmax=30,
    ts_args=dict(ylim=dict(hbo=[-15, 15], hbr=[-15, 15])),
    )
    
    # Save the plot if save is True
    if save:
        filename = f"{epoch_type}_epochs_plot.pdf"
        plt.savefig(filename)
        print(f"Plot saved as {filename}")