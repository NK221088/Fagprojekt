from fnirs_processing import epochs, tapping, control



import numpy as np
def MeanModel(TappingTest, ControlTest, TappingTrain, ControlTrain, jointArray):
    confusionMatrix = np.zeros((2,2)) # 0 = control, 1 = Tapping
    #[[TP,FP]
    # [FN,TN]]
    
    meanTapping = np.mean(jointArray[TappingTrain]) #Finding the mean of tapping training set
    meanControl = np.mean(jointArray[ControlTrain]) #Finding the mean of control training set
    
    for val in np.concatenate((TappingTest,ControlTest), axis = 0): #Iterating over values belonging to training set (both tapping and control)
            epochMean = np.mean(jointArray[val,:,:]) #Finding the mean of datapoint belonging to test set
            
            if (meanTapping - epochMean)**2 < (meanControl - epochMean)**2: #TRUE if squared distance is closer to meanTapping then meanControl
                if val < 55:                                                #Evaluating whether datapoint was labeled correctly or comitting type II error
                    confusionMatrix[0,0] += 1 
                else:
                    confusionMatrix[0,1] += 1
                    
            elif (meanTapping - epochMean)**2 > (meanControl - epochMean)**2: #TRUE if squared distance is closer to meanControl then meanTapping
                if val > 55:                                                  #Evaluating whether datapoint was labeled correctly or comitting type I error
                    confusionMatrix[1,1] += 1
                else:
                    confusionMatrix[1,0] += 1
        
    accuracy = (confusionMatrix[0,0] + confusionMatrix[1,1])/np.sum(confusionMatrix) #Calculating accuracy of current fold (TP + TN)/(N)
    
    return accuracy, confusionMatrix

def BaselineModel(TappingTest, ControlTest, TappingTrain, ControlTrain):
    testJoint = np.concatenate((TappingTest,ControlTest))
    trainJoint = np.concatenate((TappingTrain,ControlTrain))
    
    majorityClass = np.sum(55 > trainJoint) < np.sum(55 < trainJoint)
    
    if majorityClass:
        return np.sum(55 < testJoint) / len(testJoint)
    else:
        return np.sum(55 > testJoint) / len(testJoint)
    

    
    
def StratifiedCV(tappingArray, controlArray, startTime, stopTime, K = 4, freq = 7.81):
    
    baselineAccuracy_list = [] #List to store accuracies
    meanModelAccuracy_list = []
    

    dimTappingArray = tappingArray.shape[0] #Amount of tapping epochs 
    dimControlArray = controlArray.shape[0] #Amount of control epochs
    
    jointArray = np.concatenate((tappingArray,controlArray),axis = 0) #Tapping and control array stacked along the epoch-axis
    jointArray = jointArray[:,:,int(np.floor(startTime * freq)):int(np.floor(stopTime * freq))] #Remove data which are not in specified time interval
    
    randIndTapping = np.random.choice(dimTappingArray, dimTappingArray,replace=False) #Generating list of random indecies for tapping data (0, dimTappingArray)
    randIndControl = np.random.choice(dimControlArray, dimControlArray,replace=False) #Generating list of random indecies for control data (0, dimControlArray)
    randIndControl += dimTappingArray #Shifting random indecies to acount for jointArray (dimTappingArray, dimControlArray + dimTappingArray)
   
    
    
    kernelTapping = int(np.ceil(dimTappingArray/K)) #Specifing length of kernel tapping data
    kernelControl = int(np.ceil(dimControlArray/K)) #Specifing length of kernel for control data
    
    k0_tapping = 0 #First index of tapping kernel (is updated after each iteration in loop)
    k1_tapping = kernelTapping #Last index of tapping kernel (is updated after each iteration in loop)
    
    k0_control = 0 #First index of control kernel (is updated after each iteration in loop)
    k1_control = kernelControl #Last index of tapping kernel (is updated after each iteration in loop)
    
    for i in range(K):
        
        if k1_tapping > dimTappingArray: #Cutting last index of tapping kernel if too long
            k1_tapping = dimTappingArray 
        
        if k1_control > dimControlArray: #Cutting last index of control kernel if too long
            k1_control = dimControlArray
        
        
        kernelTappingTest = randIndTapping[k0_tapping:k1_tapping] #Selecting tapping kernel indecies (test data)  
        kernelTappingTrain = np.concatenate((randIndTapping[:k0_tapping],randIndTapping[k1_tapping:])) #Selecting all indecies outside of kernel (train data)
    
        kernelControlTest = randIndControl[k0_control:k1_control] #Selecting control kernel indecies (test data)  
        kernelControlTrain = np.concatenate((randIndControl[:k0_control],randIndControl[k1_control:])) #Selecting control kernel indecies (train data)

        meanModel_accuracy,_ = MeanModel(TappingTest = kernelTappingTest, ControlTest= kernelControlTest, TappingTrain = kernelTappingTrain, ControlTrain= kernelControlTrain, jointArray=jointArray)
        baseline_accuracy = BaselineModel(TappingTest = kernelTappingTest, ControlTest= kernelControlTest, TappingTrain = kernelTappingTrain, ControlTrain= kernelControlTrain)
        meanModelAccuracy_list.append(meanModel_accuracy)
        baselineAccuracy_list.append(baseline_accuracy)
    
        
        k0_tapping += kernelTapping #Updating kernel.
        k1_tapping += kernelTapping
        
        k0_control += kernelControl
        k1_control += kernelControl
        
    return np.mean(meanModelAccuracy_list), np.mean(baselineAccuracy_list)
        

    
print(StratifiedCV(epochs["Tapping"].get_data(),epochs["Control"].get_data(),startTime = 9, stopTime = 11))