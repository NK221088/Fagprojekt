from load_data_function import load_data
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

############################
# Settings:
############################

# Data set:
data_set = "fNirs_motor_full_data"
epoch_type = "Tapping"
combine_strategy = "mean"

# Data processing:
bad_channels_strategy = "all"
short_channel_correction = True
negative_correlation_enhancement = True
threshold = 3
startTime = 7.5
stopTime = 12.5
K = 3

# Plotting and saving:
plot_epochs = False
plot_std_fNIRS_response = False
plot_accuracy_across_k_folds = True

save_plot_epochs = True
save_plot_std_fNIRS_response = False
save_plot_accuracy_across_k_folds = True
save_results = True

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
plt.figure(figsize=(8, 6))
# Plot data points for each class separately and add them to the legend
plt.scatter(X_pca[labels == 0, 0], X_pca[labels == 0, 1], c='red', alpha=0.8, label=data_types[0])
plt.scatter(X_pca[labels == 1, 0], X_pca[labels == 1, 1], c='blue', alpha=0.8, label=data_types[1])

# Add legend
plt.legend(loc='upper right')

plt.title('Data Projected onto First Two Principal Components with Class Colors')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.show()