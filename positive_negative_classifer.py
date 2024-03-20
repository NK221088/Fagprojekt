import numpy as np
def Positive_Negativ_classifier(TappingTest, ControlTest, TappingTrain, ControlTrain, jointArray, labelIndx):
    confusionMatrix = np.zeros((2,2)) # 0 = control, 1 = Tapping
    #[[TP,FP]
    # [FN,TN]]
    
    
    for val in np.concatenate((TappingTest,ControlTest), axis = 0): #Iterating over values belonging to training set (both tapping and control)
            epochMean = np.mean(jointArray[val,:,:]) #Finding the mean of datapoint belonging to test set
            
            if epochMean > 0: #TRUE if squared distance is closer to meanTapping then meanControl
                if val < labelIndx:                                                #Evaluating whether datapoint was labeled correctly or comitting type II error
                    confusionMatrix[0,0] += 1 
                else:
                    confusionMatrix[0,1] += 1
                    
            elif epochMean < 0: #TRUE if squared distance is closer to meanControl then meanTapping
                if val > labelIndx:                                                  #Evaluating whether datapoint was labeled correctly or comitting type I error
                    confusionMatrix[1,1] += 1
                else:
                    confusionMatrix[1,0] += 1
        
    accuracy = (confusionMatrix[0,0] + confusionMatrix[1,1])/np.sum(confusionMatrix) #Calculating accuracy of current fold (TP + TN)/(N)
    
    return accuracy, confusionMatrix