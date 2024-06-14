from load_data_function import load_data
from sklearn.decomposition import FastICA
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from sklearn.preprocessing import StandardScaler

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

startTime = 2.5
stopTime = 10
X_1 = all_data[data_types[0]]
X_2 = all_data[data_types[1]]

X_1_new_time = X_1[:,:,int(np.floor(startTime * freq)):int(np.floor(stopTime * freq))] #Remove data which are not in specified time interval
X_2_new_time = X_2[:,:,int(np.floor(startTime * freq)):int(np.floor(stopTime * freq))] #Remove data which are not in specified time interval

# Downsample the data by taking the median of every 5 observations along the third dimension
# def downsample_median(data, window_size=60):
#     # Get the shape of the original data
#     original_shape = data.shape
    
#     # Truncate the third dimension to make it divisible by window_size
#     truncated_size = (original_shape[2] // window_size) * window_size
#     truncated_data = data[:, :, :truncated_size]
    
#     # Calculate the new shape after downsampling
#     new_shape = (original_shape[0], original_shape[1], truncated_size // window_size)
    
#     # Reshape the data to group every 5 observations
#     reshaped_data = truncated_data.reshape(original_shape[0], original_shape[1], new_shape[2], window_size)
    
#     # Take the median along the new grouping dimension
#     downsampled_data = np.median(reshaped_data, axis=3)
    
#     return downsampled_data

# Apply the downsampling function to the data
# X_1_downsampled = downsample_median(X_1_new_time)
# X_2_downsampled = downsample_median(X_2_new_time)


# X_1_reshaped = X_1_new_time.reshape(-1, all_data[data_types[0]].shape[1]) #Tapping
# X_2_reshaped = X_2_new_time.reshape(-1, all_data[data_types[1]].shape[1]) #Control

X_1_reshaped = np.median(X_1_new_time, axis=2) #Tapping
X_2_reshaped = np.median(X_2_new_time, axis=2) #Control

# scaler_x1 = StandardScaler()
# X1_normalized = scaler_x1.fit_transform(X_1_reshaped)
# scaler_x2 = StandardScaler()
# X2_normalized = scaler_x1.fit_transform(X_2_reshaped)

# Combine the data from both classes
X_combined = np.vstack((X_1_reshaped, X_2_reshaped))

scaler = StandardScaler()
X_normalized = scaler.fit_transform(X_combined)


# mean = np.mean(X_combined, axis=1, keepdims=True)
# std = np.std(X_combined, axis=1, keepdims=True)
# X_normalized = (X_combined - mean) / std

# Create labels for the classes
labels = np.hstack((np.ones(X_1_reshaped.shape[0]), np.zeros(X_2_reshaped.shape[0])))

# Initialize ICA with desired number of components
n_components = 2
ica = FastICA(n_components=n_components)

# Fit ICA to the combined data
ica.fit(X_normalized)

# Transform the combined data into the independent component space
X_ica = ica.transform(X_normalized)

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
