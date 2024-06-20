from load_data_function import load_data
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from seed import set_seeds
set_seeds()

############################
# Settings:
############################

# Data set:
data_set = "fNirs_motor_full_data"

# Data processing:
short_channel_correction = True
negative_correlation_enhancement = True

# Plotting and saving:

############################

all_epochs, data_name, all_data, freq, data_types = load_data(data_set=data_set, short_channel_correction=short_channel_correction, negative_correlation_enhancement=negative_correlation_enhancement)

X_1 = all_data[data_types[0]][-4, :, :]
X_2 = all_data[data_types[1]][-4, :, :]

# Combine the data from both classes
X_combined = np.vstack((X_1, X_2))

# Create labels for the classes
labels = np.hstack((np.zeros(X_1.shape[0]), np.ones(X_2.shape[0])))

# Initialize PCA with desired number of components
n_components = 2
pca = PCA(n_components=n_components)

# Fit PCA to the combined data
pca.fit(X_combined)

# Transform the combined data into the principal component space
X_pca = pca.transform(X_combined)

# Create a figure with a GridSpec layout
fig = plt.figure(figsize=(10, 8))
gs = GridSpec(4, 4, fig)

# Main scatter plot
ax_main = fig.add_subplot(gs[1:4, 0:3])
scatter = ax_main.scatter(X_pca[labels == 0, 0], X_pca[labels == 0, 1], c='red', alpha=0.8, label=data_types[0])
scatter = ax_main.scatter(X_pca[labels == 1, 0], X_pca[labels == 1, 1], c='blue', alpha=0.8, label=data_types[1])
ax_main.set_xlabel('Principal Component 1')
ax_main.set_ylabel('Principal Component 2')
ax_main.legend(loc='upper right')
ax_main.set_title('Data Projected onto Principal Components 0 and 1 with Class Colors')
ax_main.grid(True)

# Density plot for the x-axis (Principal Component 1)
ax_x_density = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
sns.kdeplot(X_pca[labels == 0, 0], ax=ax_x_density, color='red', fill=True, alpha=0.3)
sns.kdeplot(X_pca[labels == 1, 0], ax=ax_x_density, color='blue', fill=True, alpha=0.3)
ax_x_density.axis('off')

# Density plot for the y-axis (Principal Component 2)
ax_y_density = fig.add_subplot(gs[1:4, 3], sharey=ax_main)
sns.kdeplot(X_pca[labels == 0, 1], ax=ax_y_density, color='red', fill=True, alpha=0.3, vertical=True)
sns.kdeplot(X_pca[labels == 1, 1], ax=ax_y_density, color='blue', fill=True, alpha=0.3, vertical=True)
ax_y_density.axis('off')

# Save the plot as a PDF file
plt.savefig('pca_plot.pdf', format='pdf')

plt.show()