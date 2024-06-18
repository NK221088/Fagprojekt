import numpy as np
def MeanModel(Xtrain, ytrain, Xtest, ytest, theta):
    
    if Xtrain.ndim == 3:
        ytrain = np.array(ytrain, dtype=bool)
        ytest = np.array(ytest, dtype=bool)
        meanTapping = np.mean(Xtrain[ytrain]) #Finding the mean of tapping training set
        meanControl = np.mean(Xtrain[1 - ytrain]) #Finding the mean of control training set
        
        decision_boundary = (meanTapping + meanControl) / 2
        
        predictions = (np.mean(Xtest, axis = (1,2)) > decision_boundary) == ytest
        
    
        accuracy = sum(predictions) / len(predictions)
        
        return accuracy
    
    elif Xtrain.ndim == 2:
        
        meanTapping = np.mean(Xtrain[ytrain], axis = 0)
        meanControl = np.mean(Xtrain[1 - ytrain], axis = 0)
        
        predictions = []
        
        for point in Xtest:
            if np.linalg.norm(point - meanTapping) < np.linalg.norm(point - meanControl):
                predictions.append(True)
            else:
                predictions.append(False)
                
        accuracy = sum(np.array(predictions) == ytest) / len(predictions)
        
        return accuracy
        
        
        
        