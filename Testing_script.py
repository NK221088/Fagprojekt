# Script only used for running classifiers and tests
from stratified_cv import StratifiedCV
from majority_voting_classifier import BaselineModel
from mean_model_classifier import MeanModel
from fnirs_processing import all_epochs, epochs, all_tapping, all_control, all_epochs_con
from epoch_plot import epoch_plot
import mne

epoch_plot(all_epochs[0], type = "Tapping", combine_strategy = "mean")
print(StratifiedCV(all_tapping,all_control,startTime = 5, K=3, stopTime = 11, freq=3.90625))