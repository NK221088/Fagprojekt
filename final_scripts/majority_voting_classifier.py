import numpy as np
from seed import set_seeds
set_seeds()

def BaselineModel(Xtrain, ytrain, Xtest, ytest, theta):
    N = len(ytest)
    
    # Predict the majority class from the training set
    if sum(ytrain == 1) < sum(ytrain == 0):
        y_pred = np.zeros_like(ytest)  # Predict all zeros
    else:
        y_pred = np.ones_like(ytest)   # Predict all ones
    
    # Calculate accuracy
    accuracy = np.sum(y_pred == ytest) / N
    
    # Calculate confusion matrix components
    tp = np.sum((y_pred == 1) & (ytest == 1))  # True Positives
    tn = np.sum((y_pred == 0) & (ytest == 0))  # True Negatives
    fp = np.sum((y_pred == 1) & (ytest == 0))  # False Positives
    fn = np.sum((y_pred == 0) & (ytest == 1))  # False Negatives
    
    # Create confusion matrix
    cm = np.array([[tn, fp],
                   [fn, tp]])
    
    return accuracy, cm