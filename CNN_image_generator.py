import os
import matplotlib.pyplot as plt
from load_data_function import load_data
from sklearn.model_selection import train_test_split
import numpy as np


def create_directories(base_dir, class_names, sub_dirs):
    paths = {}
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    for class_name in class_names:
        class_paths = {}
        for sub_dir in sub_dirs:
            dir_path = os.path.join(base_dir, sub_dir, class_name)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            class_paths[sub_dir] = dir_path
        paths[class_name] = class_paths
    return paths


# Define base directory and class names
base_dir = 'fNIRS_images'
class_names = ['Tapping', 'Control']
sub_dirs = ['train', 'val']

# Create directories and retrieve paths
paths = create_directories(base_dir, class_names, sub_dirs)


def save_fNIRS_data_as_images(data_dict, paths, split_ratio=0.8):
    for class_name, data in data_dict.items():
        # Create labels for stratification: Array of same length as data, filled with the class_name
        labels = np.array([class_name] * len(data))

        # Split data into training and validation sets with stratification
        train_data, val_data = train_test_split(data, train_size=split_ratio, random_state=42, stratify=labels)

        # Save training data
        for i, epoch_data in enumerate(train_data):
            file_path = os.path.join(paths[class_name]['train'], f'{class_name}_{i}.png')
            plt.imsave(file_path, epoch_data, cmap='gray')

        # Save validation data
        for i, epoch_data in enumerate(val_data):
            file_path = os.path.join(paths[class_name]['val'], f'{class_name}_{i}.png')
            plt.imsave(file_path, epoch_data, cmap='gray')


# Example of loading data
all_epochs, data_name, all_data, freq, data_types = load_data(data_set="fNirs_motor_full_data",
                                                              short_channel_correction=True,
                                                              negative_correlation_enhancement=True)

# Save images to respective directories
save_fNIRS_data_as_images(all_data, paths)