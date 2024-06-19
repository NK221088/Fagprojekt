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
        self.n = 2
        self.mask = ()
        self.useMask = False
        
        self.useICA = False

        
        if self.name == "ANN":
            self.gaussian_bound = {'neurons1': (50,70), 'neurons2': (50,200), 'layers': (0.51,3.49), 'learning_rate': (0.51,2.49)}
        elif self.name == 'SVM':
            self.gaussian_bound = {'kernel': (0.5,4.5)}
        elif self.name == 'CNN':
            self.gaussian_bound = {'learning_rate': (0.0001, 0.01)}
        elif self.name == "Mean":
            self.gaussian_bound = {}
           
        
        
        if theta == None:
            if self.name == "SVM":
                self.theta = {'kernel': ["linear", "poly", "rbf", "sigmoid"]}
            elif self.name == "ANN":
                self.theta = {"neurons1": [50, 60, 70], "neurons2": [50, 100, 150, 200], "layers" : [4,6], "learning_rate": ["decrease", "clr"]}
            elif self.name == "Mean":
                self.theta = []
            elif self.name == "Baseline":
                self.theta = []
            elif self.name == "PosNeg":
                self.theta = []
            elif self.name == "CNN":
                self.theta = {'learning_rate': [0.0001,0.001,0.01]}
        
        
    def train(self, theta):
        
        if self.name == "SVM":
            if self.useMask:
                return SVM_classifier(Xtrain = self.Xtrain[:,self.mask], ytrain = self.ytrain, Xtest = self.Xtest[:,self.mask], ytest = self.ytest, theta = theta)
            else:
                return SVM_classifier(Xtrain = self.Xtrain, ytrain = self.ytrain, Xtest = self.Xtest, ytest = self.ytest, theta = theta)
        elif self.name == "ANN":
            if self.useMask:
                return ANN_classifier(Xtrain = self.Xtrain[:,self.mask], ytrain = self.ytrain, Xtest = self.Xtest[:,self.mask], ytest = self.ytest, theta = theta)
            else:
                return ANN_classifier(Xtrain = self.Xtrain, ytrain = self.ytrain, Xtest = self.Xtest, ytest = self.ytest, theta = theta)
        elif self.name == "Mean":
            return MeanModel(Xtrain = self.Xtrain, ytrain = self.ytrain, Xtest = self.Xtest, ytest = self.ytest, theta = theta)        
        elif self.name == "Baseline":
            return BaselineModel(Xtrain = self.Xtrain, ytrain = self.ytrain, Xtest = self.Xtest, ytest = self.ytest, theta = theta)
        elif self.name == "PosNeg":
            return Positive_Negative_classifier(Xtrain = self.Xtrain, ytrain = self.ytrain, Xtest = self.Xtest, ytest = self.ytest, theta = theta)
        elif self.name == "CNN":
            if self.useMask:
                return CNN_classifier(Xtrain = self.Xtrain[:,self.mask], ytrain = self.ytrain, Xtest = self.Xtest[:,self.mask], ytest = self.ytest, theta = theta)[1]
            else:
                return CNN_classifier(Xtrain = self.Xtrain, ytrain = self.ytrain, Xtest = self.Xtest, ytest = self.ytest, theta = theta)[1]
        
    def getTheta(self):
        return self.theta
    
    def load(self, Xtrain, Xtest, ytrain, ytest):
        self.Xtrain = Xtrain
        self.Xtest = Xtest
        self.ytrain = ytrain
        self.ytest = ytest
        

        
    def objective_function(self, bayes = True, **kwargs):
        
        #Loading data with selected features
        self.useMask = True
        self.mask = tuple(i for i in range(self.n) if int(np.rint(kwargs[f'Feature_{i}'])) == 1)
        
        if bayes:
            if self.name == 'ANN':
                
                layer_dic = {1: 4, 2: 6, 3: 8}
                
                if int(np.rint(kwargs['learning_rate'])) == 1:
                    kwargs['learning_rate'] = 'decrease'
                    
                elif int(np.rint(kwargs['learning_rate'])) == 2:
                    kwargs['learning_rate'] = 'clr'
                
                
                kwargs['layers'] = layer_dic[int(np.rint(kwargs['layers']))]
                    
                kwargs['neurons1'] = int(np.rint(kwargs['neurons1']))
                
                kwargs['neurons2'] = int(np.rint(kwargs['neurons2']))
                
                    
                return self.train(kwargs)
            
            elif self.name == 'SVM':
                if int(np.rint(kwargs['kernel'])) == 1:
                    kwargs['kernel'] = 'linear'
                    
                elif int(np.rint(kwargs['kernel'])) == 2:
                    kwargs['kernel'] = 'poly'
                    
                elif int(np.rint(kwargs['kernel'])) == 3:
                    kwargs['kernel'] = 'rbf'
                    
                elif int(np.rint(kwargs['kernel'])) == 4:
                    kwargs['kernel'] = 'sigmoid'
                    
                return self.train(kwargs)
            
            
            return self.train(kwargs)
        else:
            if kwargs['Feature_0'] == 0 and kwargs['Feature_1'] == 0:
                return 0
            return self.train(kwargs)

        
        

        
        