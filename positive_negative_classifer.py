import numpy as np
def Positive_Negativ_classifier(Xtrain, ytrain, Xtest, ytest):
    
    predictions = (np.mean(Xtest, axis = (1,2)) > 0) == ytest
    
    accuracy = sum(predictions) / len(predictions)
    
    return accuracy