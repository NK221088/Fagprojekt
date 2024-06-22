import numpy as np
from seed import set_seeds
from sklearn.metrics import confusion_matrix

set_seeds()

def Positive_Negative_classifier(Xtrain, ytrain, Xtest, ytest, theta):
    
    predictions = (np.mean(Xtest, axis=(1,2)) > 0)
    
    accuracy = sum(predictions == ytest) / len(ytest)
    
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(ytest, predictions)
    
    return accuracy, conf_matrix, (predictions, ytest)
