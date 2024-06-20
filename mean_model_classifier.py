import numpy as np

import tensorflow as tf
import numpy as np
import random
import os

# Set seeds for reproducibility
def set_seeds(seed=42):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # If using TensorFlow with GPU:
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

set_seeds()

def MeanModel(Xtrain, ytrain, Xtest, ytest, theta):
    ytrain = np.array(ytrain, dtype=bool)
    ytest = np.array(ytest, dtype=bool)
    meanTapping = np.mean(Xtrain[ytrain]) #Finding the mean of tapping training set
    meanControl = np.mean(Xtrain[1 - ytrain]) #Finding the mean of control training set
    
    decision_boundary = (meanTapping + meanControl) / 2
    
    predictions = (np.mean(Xtest, axis = (1,2)) > decision_boundary) == ytest
    
  
    accuracy = sum(predictions) / len(predictions)
    
    return accuracy