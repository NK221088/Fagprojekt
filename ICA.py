from load_data_function import load_data
from sklearn.decomposition import FastICA
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import os
from datetime import datetime

def ICA(Xtrain , Xtest, n_components: int, plot: bool, save_plot: bool, components: tuple = (0, 1)):
    """
    Perform Independent Component Analysis (ICA) on the given data and optionally plot the results.

    Parameters:
    - all_data (dict): Dictionary containing the data for different types.
    - data_types (list): List of data type keys to be used from the all_data dictionary.
    - n_components (int): Number of independent components to compute.
    - plot (bool): Whether to plot the data projected onto the first two independent components.
    - save_plot (bool): Whether to save the plot.
    - components (tuple): Tuple of two integers specifying which components to plot.

    Returns:
    - X_ica (numpy.ndarray): The data transformed into the independent component space.
    """
    
    Xtrain = Xtrain.reshape(Xtrain.shape[0], -1)
    Xtest = Xtest.reshape(Xtest.shape[0], -1)
    # Initialize ICA with desired number of components
    ica = FastICA(n_components=n_components)

    Xtrain_ica = ica.fit_transform(Xtrain)
    Xtest_ica = ica.transform(Xtest)
    return Xtrain_ica, Xtest_ica
"""    
    if plot or save_plot:
        comp1, comp2 = components
        
        # Create a figure with a GridSpec layout
        fig = plt.figure(figsize=(10, 8))
        gs = GridSpec(4, 4, fig)

        # Main scatter plot
        ax_main = fig.add_subplot(gs[1:4, 0:3])
        scatter = ax_main.scatter(X_ica[labels == 0, comp1], X_ica[labels == 0, comp2], c='red', alpha=0.8, label=data_types[0])
        scatter = ax_main.scatter(X_ica[labels == 1, comp1], X_ica[labels == 1, comp2], c='blue', alpha=0.8, label=data_types[1])
        ax_main.set_xlabel(f'Independent Component {comp1 + 1}')
        ax_main.set_ylabel(f'Independent Component {comp2 + 1}')
        ax_main.legend(loc='upper right')
        ax_main.set_title(f'Data Projected onto Independent Components {comp1 + 1} and {comp2 + 1} with Class Colors')
        ax_main.grid(True)

        # Density plot for the x-axis (Independent Component comp1)
        ax_x_density = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
        sns.kdeplot(X_ica[labels == 0, comp1], ax=ax_x_density, color='red', fill=True, alpha=0.3)
        sns.kdeplot(X_ica[labels == 1, comp1], ax=ax_x_density, color='blue', fill=True, alpha=0.3)
        ax_x_density.axis('off')

        # Density plot for the y-axis (Independent Component comp2)
        ax_y_density = fig.add_subplot(gs[1:4, 3], sharey=ax_main)
        sns.kdeplot(X_ica[labels == 0, comp2], ax=ax_y_density, color='red', fill=True, alpha=0.3, vertical=True)
        sns.kdeplot(X_ica[labels == 1, comp2], ax=ax_y_density, color='blue', fill=True, alpha=0.3, vertical=True)
        ax_y_density.axis('off')
        
        # Show plot if requested
        if plot:
            plt.show()

        # Save plot if requested
        if save_plot:
            plots_folder = "Plots"
            os.makedirs(plots_folder, exist_ok=True)
            current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = os.path.join(plots_folder, f"ICA_plot_Components_{comp1+1}_{comp2+1}_{current_datetime}.pdf")
            fig.savefig(filename)
            print(f"Plot saved as {filename}")
            plt.close(fig)  # Close the figure after saving
"""
