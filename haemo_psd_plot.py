from Participant_class import individual_participant_class
from datetime import datetime

import tensorflow as tf
import numpy as np
import random
import os

# Set seeds for reproducibility
def set_seeds(seed=42):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # If using TensorFlow with GPU:
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

set_seeds()

def plot_psd_individual(individual, save=False):
    """
    Plot Power Spectral Density (PSD) of raw haemo data for an individual participant before and after filtering.

    Parameters:
    - individual (individual_participant_class): An instance of the individual_participant_class representing the participant whose data will be plotted.
    - save (bool): Flag indicating whether to save the plot. Default is False.

    Returns:
    None
    """
    import matplotlib.pyplot as plt

    # Check if individual is of the correct type
    if not isinstance(individual, individual_participant_class):
        raise TypeError("individual must be an instance of individual_participant_class")

    # Fixed title for the after filtering plot
    after_filter_title = "After filtering"

    # Plot PSD before filtering
    fig_before = individual.raw_haemo_unfiltered.plot_psd(average=True)
    fig_before.suptitle("Before filtering", weight='bold', size='x-large')
    fig_before.subplots_adjust(top=0.88)

    # Filter the raw haemo data
    if individual.raw_haemo_unfiltered is not None:
        raw_haemo_filtered = individual.raw_haemo_unfiltered.copy().filter(0.05, 0.7, h_trans_bandwidth=0.2, l_trans_bandwidth=0.02)

        # Plot PSD after filtering
        fig_after = raw_haemo_filtered.plot_psd(average=True)
        fig_after.suptitle(after_filter_title, weight='bold', size='x-large')
        fig_after.subplots_adjust(top=0.88)

        # Save the plot if specified
        if save:
            current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            plots_folder = "Plots"
            filename_before = os.path.join(plots_folder, f"{individual.name}_PSD_before_filtering_{current_datetime}.pdf")
            filename_after = os.path.join(plots_folder, f"{individual.name}_PSD_after_filtering_{current_datetime}.pdf")
            fig_before.savefig(filename_before)
            fig_after.savefig(filename_after)
            print(f"Plots saved as {filename_before} and {filename_after}")
            plt.close(fig_before)
            plt.close(fig_after)
    else:
        print("No raw_haemo_unfiltered data available for plotting.")

