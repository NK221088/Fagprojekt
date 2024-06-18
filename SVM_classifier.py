from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

def SVM_classifier(Xtrain, ytrain, Xtest, ytest, theta):
    
    clf = svm.SVC(kernel= theta["kernel"], C = theta["C"], gamma= theta["gamma"], degree= theta["degree"], coef0= theta["coef0"])

    
    Xtrain = Xtrain.reshape(Xtrain.shape[0], -1)
    Xtest = Xtest.reshape(Xtest.shape[0], -1)
    
    clf.fit(Xtrain, ytrain)

    accuracy = clf.score(X = Xtest ,y = ytest)

    return accuracy