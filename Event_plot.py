from datetime import datetime
import mne
import os
import numpy as np
from Participant_class import individual_participant_class
import matplotlib.pyplot as plt
from haemo_psd_plot import plot_psd_individual

def event_plot(individual_to_plot: individual_participant_class, type_of_event_plot: str, save: bool = False):
    """
    Plot event-related data for a given individual participant.

    Parameters:
    - individual_to_plot (individual_participant_class): An instance of the individual_participant_class representing the participant whose data will be plotted.
    - type_of_event_plot (str): The type of event-related data to plot. Supported values are 'raw_intensity', 'raw_od', and 'raw_haemo'.
    - save (bool): Flag indicating whether to save the plot. Default is False.

    Returns:
    None

    Raises:
    - ValueError: If the type_of_event_plot is not one of the supported values.
    - TypeError: If individual_to_plot is not an instance of individual_participant_class.
    """
    # Check if individual_to_plot is of the correct type
    if not isinstance(individual_to_plot, individual_participant_class):
        raise TypeError("individual_to_plot must be an instance of individual_participant_class")

    # Check if type_of_event_plot is one of the supported values
    if type_of_event_plot not in ['raw_intensity', 'raw_od', 'raw_haemo']:
        raise ValueError("type_of_event_plot must be one of 'raw_intensity', 'raw_od', or 'raw_haemo'")

    # Plot the requested data type
    if type_of_event_plot == "raw_intensity":
        if individual_to_plot.raw_intensity is not None:
            plot = individual_to_plot.raw_intensity.plot(n_channels=len(individual_to_plot.raw_intensity.ch_names),
                                                         duration=500, show_scrollbars=False)
        else:
            print("No raw_intensity data available for plotting.")
    elif type_of_event_plot == "raw_od":
        if individual_to_plot.raw_od is not None:
            plot = individual_to_plot.raw_od.plot(n_channels=len(individual_to_plot.raw_od.ch_names),
                                                  duration=500, show_scrollbars=False)
        else:
            print("No raw_od data available for plotting.")
    elif type_of_event_plot == "raw_haemo":
        if individual_to_plot.raw_haemo is not None:
            plot = individual_to_plot.raw_haemo.plot(n_channels=len(individual_to_plot.raw_haemo.ch_names),
                                                      duration=500, show_scrollbars=False)
        else:
            print("No raw_haemo data available for plotting.")

# Save the plot if specified
    if save:
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plots_folder = "Plots"
        filename = os.path.join(plots_folder, f"{individual_to_plot.name}_{type_of_event_plot}_plot_{current_datetime}.pdf")
        plt.savefig(filename)  # Save the plot
        print(f"Plot saved as {filename}")
        plt.close()  # Close the figure after saving