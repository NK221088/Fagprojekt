from load_data_function import load_data
from two_level_cross_validation import two_level_cross_validation
import mne
import os
from collections import Counter
from model import model
import numpy as np

############################
# Settings:
############################

# Data set:
data_set = "fNirs_motor_full_data"
epoch_type = "Tapping"
combine_strategy = "mean"
individuals = True

# Data processing:
bad_channels_strategy = "all"
short_channel_correction = True
negative_correlation_enhancement = True
threshold = 3
startTime = 7.5
stopTime = 12.5
K = 10

#Models
SVM = model(name = "SVM")
ANN = model(name = "ANN")

modelList = [SVM, ANN]


all_epochs, data_name, all_data, freq, data_types = load_data(data_set = data_set, short_channel_correction = short_channel_correction, negative_correlation_enhancement = negative_correlation_enhancement, individuals = individuals)
two_level_cross_validation(modelList = modelList, )


