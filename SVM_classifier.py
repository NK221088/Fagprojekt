from sklearn import svm
import numpy as np

def SVM_classifier(TappingTest, ControlTest, TappingTrain, ControlTrain, jointArray, labelIndx):
    
    
    
    confusionMatrix = np.zeros((2,2)) # 0 = control, 1 = Tapping
    #[[TP,FP]
    # [FN,TN]]
    clf = svm.SVC()
    X = jointArray[np.concatenate((TappingTrain, ControlTrain)),:,:]
    y = np.concatenate((np.ones(len(TappingTrain)), np.zeros(ControlTrain)))
    
    clf.fit(X,y)
    
    for val in np.concatenate((TappingTest,ControlTest), axis = 0): #Iterating over values belonging to training set (both tapping and control)
            
            clf.predict(jointArray[val,:,:])
        
        
    accuracy = (confusionMatrix[0,0] + confusionMatrix[1,1])/np.sum(confusionMatrix) #Calculating accuracy of current fold (TP + TN)/(N)
    
    return accuracy, confusionMatrix

