# Script only used for running classifiers and tests
from stratified_cv import StratifiedCV
from majority_voting_classifier import BaselineModel
from mean_model_classifier import MeanModel
from fnirs_processing import epochs, tapping, control


print(StratifiedCV(epochs["Tapping"].get_data(),epochs["Control"].get_data(),startTime = 9, stopTime = 11))