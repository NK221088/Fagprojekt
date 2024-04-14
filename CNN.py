from datetime import datetime
from stratified_cv import StratifiedCV
from majority_voting_classifier import BaselineModel
from mean_model_classifier import MeanModel
from load_data_function import load_data
from epoch_plot import epoch_plot
from standard_fNIRS_response_plot import standard_fNIRS_response_plot
from Plotting_of_accuracy_across_k_folds import plot_of_accuracy_across_k_folds
import mne
import os
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

############################
#Settings:
############################

#Data set:
data_set = "fNirs_motor_full_data"


short_channel_correction = True
negative_correlation_enhancement = True

############################

all_epochs, data_name, all_data, freq, data_types = load_data(data_set = data_set, short_channel_correction = short_channel_correction, negative_correlation_enhancement = negative_correlation_enhancement)

plt.imshow(all_data["Control"][4], cmap='Grays', interpolation='nearest')

plt.show()

plt.imshow(all_data["Tapping"][4], cmap='Grays', interpolation='nearest')

plt.show()