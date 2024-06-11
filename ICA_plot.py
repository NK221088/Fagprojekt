from load_data_function import load_data
from sklearn.decomposition import FastICA
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

############################
# Settings:
############################

# Data set:
data_set = "fNirs_motor_full_data"

# Data processing:
short_channel_correction = True
negative_correlation_enhancement = True

############################

all_epochs, data_name, all_data, freq, data_types = load_data(data_set=data_set, short_channel_correction=short_channel_correction, negative_correlation_enhancement=negative_correlation_enhancement)

X_1 = all_data[data_types[0]].reshape(all_data[data_types[0]].shape[0], -1)
X_2 = all_data[data_types[1]].reshape(all_data[data_types[1]].shape[0], -1)

# Combine the data from both classes
X_combined = np.vstack((X_1, X_2))
mean = np.mean(X_combined, axis=1, keepdims=True)
std = np.std(X_combined, axis=1, keepdims=True)
X_normalized = (X_combined - mean) / std

# Create labels for the classes
labels = np.hstack((np.zeros(X_1.shape[0]), np.ones(X_2.shape[0])))

# Initialize ICA with desired number of components
n_components = 2
ica = FastICA(n_components=n_components)

# Fit ICA to the combined data
ica.fit(X_combined)

# Transform the combined data into the independent component space
X_ica = ica.transform(X_combined)

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
ax_main.set_title('Data Projected onto Independent Components 1 and 2 with Class Colors')
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

# Save the figure as a PDF
plt.savefig('ica_projection_plot.pdf')

plt.show()
