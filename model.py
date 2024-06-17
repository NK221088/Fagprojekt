import numpy as np
from ANN import ANN_classifier
from SVM_classifier import SVM_classifier
from mean_model_classifier import MeanModel
from majority_voting_classifier import BaselineModel
from positive_negative_classifer import Positive_Negative_classifier
from CNN_pretrained_model import CNN_classifier


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
            return SVM_classifier(Xtrain = Xtrain, ytrain = ytrain, Xtest = Xtest, ytest = ytest, theta = theta)
        elif self.name == "ANN":
            return ANN_classifier(Xtrain = Xtrain, ytrain = ytrain, Xtest = Xtest, ytest = ytest, theta = theta)
        elif self.name == "Mean":
            return MeanModel(Xtrain = Xtrain, ytrain = ytrain, Xtest = Xtest, ytest = ytest, theta = theta)        
        elif self.name == "Baseline":
            return BaselineModel(Xtrain = Xtrain, ytrain = ytrain, Xtest = Xtest, ytest = ytest, theta = theta)
        elif self.name == "PosNeg":
            return Positive_Negative_classifier(Xtrain = Xtrain, ytrain = ytrain, Xtest = Xtest, ytest = ytest, theta = theta)
        elif self.name == "CNN":
            return CNN_classifier(Xtrain = Xtrain, ytrain = ytrain, Xtest = Xtest, ytest = ytest, theta = theta)[1]
        
    def getTheta(self):
        return self.theta
        
        