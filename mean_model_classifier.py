import numpy as np
def MeanModel(Xtrain, ytrain, Xtest, ytest):
    # Create boolean masks for tapping and control classes
    tapping_mask = ytrain == 1
    control_mask = ytrain == 0
    
    # Calculate mean for tapping and control classes using boolean masks
    meanTapping = np.mean(Xtrain[tapping_mask], axis=0)
    meanControl = np.mean(Xtrain[control_mask], axis=0)
    
    # Calculate decision boundary
    decision_boundary = (meanTapping + meanControl) / 2
    
    # Make predictions based on decision boundary
    predictions = (np.mean(Xtest, axis=(1, 2)) > decision_boundary) == ytest
    
    # Calculate accuracy
    accuracy = np.sum(predictions) / len(predictions)
    
    return accuracy
