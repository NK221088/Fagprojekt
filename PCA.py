from load_data_function import load_data
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import Axes3D for 3D plotting
from scipy.stats import gaussian_kde

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

all_epochs, data_name, all_data, freq, data_types = load_data(data_set = data_set, short_channel_correction = short_channel_correction, negative_correlation_enhancement = negative_correlation_enhancement)

X_1 = all_data[data_types[0]][-1, :, :]
X_2 = all_data[data_types[1]][-1, :, :]

# Combine the data from both classes
X_combined = np.vstack((X_1, X_2))

# Create labels for the classes
labels = np.hstack((np.zeros(X_1.shape[0]), np.ones(X_2.shape[0])))

# Initialize PCA with desired number of components
n_components = 3
pca = PCA(n_components=n_components)

# Fit PCA to the combined data
pca.fit(X_combined)

# Transform the combined data into the principal component space
X_pca = pca.transform(X_combined)

# Plot the data projected onto the first two principal components with colors distinguishing between the classes
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')  # Create a 3D subplot

# Plot data points for each class separately and add them to the legend
ax.scatter(X_pca[labels == 0, 0], X_pca[labels == 0, 1], X_pca[labels == 0, 2], c='red', alpha=0.8, label=data_types[0])
ax.scatter(X_pca[labels == 1, 0], X_pca[labels == 1, 1], X_pca[labels == 1, 2], c='blue', alpha=0.8, label=data_types[1])

# Add legend
plt.legend(loc='upper right')

ax.set_title('Data Projected onto First Three Principal Components with Class Colors')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
plt.grid(True)
plt.show()