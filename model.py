import numpy as np
from ANN import ANN_classifier
from SVM_classifier import SVM_classifier
from mean_model_classifier import MeanModel
from majority_voting_classifier import BaselineModel
from positive_negative_classifer import Positive_Negative_classifier
from CNN_pretrained_model import CNN_classifier

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


class model:
    def __init__(self, name, theta = None):
        
        self.name = name
        self.theta = theta
        
        if theta == None:
            if self.name == "SVM":
                self.theta = ["linear", "poly", "rbf", "sigmoid"]           
            elif self.name == "ANN":
                self.theta = {
    "neuron1": [60, 128],
    "neuron2": [100, 150, 300],
    "layers": [6, 8],
    "learning_rate": ["decrease", "clr"],
}
            elif self.name == "Mean":
                self.theta = {}
            elif self.name == "Baseline":
                self.theta = []
            elif self.name == "PosNeg":
                self.theta = []
            elif self.name == "CNN":
                self.theta = [0.0001,0.001,0.01]
        
        
    def train(self, Xtrain, ytrain, Xtest, ytest, theta):
        if self.name == "SVM":
            (accuracy, cm) = SVM_classifier(Xtrain = Xtrain, ytrain = ytrain, Xtest = Xtest, ytest = ytest, theta = theta)
            return accuracy, cm
        elif self.name == "ANN":
            (accuracy, cm) = ANN_classifier(Xtrain = Xtrain, ytrain = ytrain, Xtest = Xtest, ytest = ytest, theta = theta)
            return accuracy, cm
        elif self.name == "Mean":
            (accuracy, cm) = MeanModel(Xtrain = Xtrain, ytrain = ytrain, Xtest = Xtest, ytest = ytest, theta = theta)
            return accuracy, cm        
        elif self.name == "Baseline":
            (accuracy, cm) = BaselineModel(Xtrain = Xtrain, ytrain = ytrain, Xtest = Xtest, ytest = ytest, theta = theta)
            return accuracy, cm
        elif self.name == "PosNeg":
            (accuracy, cm) = Positive_Negative_classifier(Xtrain = Xtrain, ytrain = ytrain, Xtest = Xtest, ytest = ytest, theta = theta)
            return accuracy, cm
        elif self.name == "CNN":
            (accuracy, cm) = CNN_classifier(Xtrain = Xtrain, ytrain = ytrain, Xtest = Xtest, ytest = ytest, theta = theta)
            return accuracy, cm
        
    def getTheta(self):
        return self.theta
        
        