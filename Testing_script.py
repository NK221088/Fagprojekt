# Script only used for running classifiers and tests
from stratified_cv import StratifiedCV
from majority_voting_classifier import BaselineModel
from mean_model_classifier import MeanModel
from fnirs_processing import all_epochs, epochs, all_tapping, all_control
from epoch_plot import epoch_plot
import mne

epoch_plot(epochs, type = "Tapping", combine_strategy = "mean")
print(StratifiedCV(all_tapping,all_control,startTime = 3, K=3, stopTime = 7, freq=5.2))