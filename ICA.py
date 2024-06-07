from load_data_function import load_data
from sklearn.decomposition import FastICA
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

def ICA(all_data, data_types: list, n_components: int, plot: bool, save_plot: bool):
    
    X_1 = all_data[data_types[0]][-1, :, :]
    X_2 = all_data[data_types[1]][-1, :, :]

    # Combine the data from both classes
    X_combined = np.vstack((X_1, X_2))


    # Create labels for the classes
    labels = np.hstack((np.zeros(X_1.shape[0]), np.ones(X_2.shape[0])))

    # Initialize ICA with desired number of components
    ica = FastICA(n_components=n_components)

    # Fit ICA to the combined data
    ica.fit(X_combined)

    # Transform the combined data into the independent component space
    X_ica = ica.transform(X_combined)
    
    if plot:
        # Create a figure with a GridSpec layout
        fig = plt.figure(figsize=(10, 8))
        gs = GridSpec(4, 4, fig)

        # Main scatter plot
        ax_main = fig.add_subplot(gs[1:4, 0:3])
        scatter = ax_main.scatter(X_ica[labels == 0, 0], X_ica[labels == 0, 1], c='red', alpha=0.8, label=data_types[0])
        scatter = ax_main.scatter(X_ica[labels == 1, 0], X_ica[labels == 1, 1], c='blue', alpha=0.8, label=data_types[1])
        ax_main.set_xlabel('Independent Component 1')
        ax_main.set_ylabel('Independent Component 2')
        ax_main.legend(loc='upper right')
        ax_main.set_title('Data Projected onto Independent Components 0 and 1 with Class Colors')
        ax_main.grid(True)

        # Density plot for the x-axis (Independent Component 1)
        ax_x_density = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
        sns.kdeplot(X_ica[labels == 0, 0], ax=ax_x_density, color='red', fill=True, alpha=0.3)
        sns.kdeplot(X_ica[labels == 1, 0], ax=ax_x_density, color='blue', fill=True, alpha=0.3)
        ax_x_density.axis('off')

        # Density plot for the y-axis (Independent Component 2)
        ax_y_density = fig.add_subplot(gs[1:4, 3], sharey=ax_main)
        sns.kdeplot(X_ica[labels == 0, 1], ax=ax_y_density, color='red', fill=True, alpha=0.3, vertical=True)
        sns.kdeplot(X_ica[labels == 1, 1], ax=ax_y_density, color='blue', fill=True, alpha=0.3, vertical=True)
        ax_y_density.axis('off')
        plt.show()
        
    
    return X_ica            