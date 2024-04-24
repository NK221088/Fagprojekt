import numpy as np
def MeanModel(Xtrain, ytrain, Xtest, ytest, theta):
    
    meanTapping = np.mean(Xtrain[ytrain]) #Finding the mean of tapping training set
    meanControl = np.mean(Xtrain[1 - ytrain]) #Finding the mean of control training set
    
    decision_boundary = (meanTapping + meanControl) / 2
    
    predictions = (np.mean(Xtest, axis = (1,2)) > decision_boundary) == ytest
    
  
    accuracy = sum(predictions) / len(predictions)
    
    return accuracy