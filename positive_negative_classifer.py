import numpy as np
<<<<<<< HEAD
def Positive_Negativ_classifier(Xtrain, ytrain, Xtest, ytest):
=======
def Positive_Negative_classifier(TappingTest, ControlTest, TappingTrain, ControlTrain, jointArray, labelIndx, theta):
    confusionMatrix = np.zeros((2,2)) # 0 = control, 1 = Tapping
    #[[TP,FP]
    # [FN,TN]]
>>>>>>> 2-Level-Cross-Validation
    
    predictions = (np.mean(Xtest, axis = (1,2)) > 0) == ytest
    
    accuracy = sum(predictions) / len(predictions)
    
    return accuracy