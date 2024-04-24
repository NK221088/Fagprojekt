import numpy as np
from ANN import ANN_classifier
from SVM_classifier import SVM_classifier
from mean_model_classifier import MeanModel
from majority_voting_classifier import BaselineModel
from positive_negative_classifier import Positive_Negative_classifier


class model:
    def __init__(self, name, theta):
        self.name = name
        self.theta = theta
        
        if self.name == "SVM":
            self.theta = ["linear", "poly", "rbf", "sigmoid", "precomputed"]           
        elif self.name == "ANN":
            self.theta = np.linspace(103,303,9, dtype = int)
        elif self.name == "Mean":
            self.theta = "None"
        elif self.name == "Baseline":

        elif self.name == "PosNeg":
        
        
    def train(self, Xtrain, ytrain, Xtest, ytest):
        if self.name == "SVM":
            return SVM_classifier(Xtrain = Xtrain, ytrain = ytrain, Xtest = Xtest, ytest = ytest, theta = theta)
        elif self.name == "ANN":
            return ANN_classifier(Xtrain = Xtrain, ytrain = ytrain, Xtest = Xtest, ytest = ytest, theta = theta)
        elif self.name == "Mean":
            return MeanModel(Xtrain = Xtrain, ytrain = ytrain, Xtest = Xtest, ytest = ytest, theta = theta)        
        elif self.name == "Baseline":
            return BaselineModel(Xtrain = Xtrain, ytrain = ytrain, Xtest = Xtest, ytest = ytest, theta = theta)
        elif self.name == "PosNeg":
            return Positive_Negative_classifier(Xtrain = Xtrain, ytrain = ytrain, Xtest = Xtest, ytest = ytest, theta = theta)
        
        