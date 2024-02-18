from fnirs_processing import epochs, tapping, control



import numpy as np
def BaselineClassifier(tappingArray,controlArray, startTime, stopTime, test_train_split = 0.8, freq = 7.81):
    

    dimTappingArray = tappingArray.shape[0]
    dimControlArray = controlArray.shape[0]
    
    jointArray = np.concatenate((tappingArray,controlArray),axis = 0)
    jointArray = jointArray[:,:,int(np.floor(startTime * freq)):int(np.floor(stopTime * freq))]
    
    randIndTapping = np.random.choice(dimTappingArray, dimTappingArray,replace=False)
    randIndControl = np.random.choice(dimControlArray, dimControlArray,replace=False)
    randIndControl += dimTappingArray
    
    tappingBoundery = int(np.floor(test_train_split * dimTappingArray))
    controlBoundery = int(np.floor(test_train_split * dimControlArray))
    
    tappingTrain = randIndTapping[:tappingBoundery]
    tappingTest = randIndTapping[tappingBoundery:]
    controlTrain = randIndControl[:controlBoundery]
    controlTest = randIndControl[controlBoundery:]
    
    meanTapping = np.mean(jointArray[tappingTrain,:,:])
    meanControl = np.mean(jointArray[controlTrain,:,:])
    
    confusionMatrix = np.zeros((2,2)) # 0 = control, 1 = Tapping
    #[[TP,FP]
    # [FN,TN]]
    
    for val in np.concatenate((tappingTest,controlTest), axis = 0):
        epochMean = np.mean(jointArray[val,:,:])
        
        if (meanTapping - epochMean)**2 < (meanControl - epochMean)**2:
            if val < 55:
                confusionMatrix[0,0] += 1
            else:
                confusionMatrix[0,1] += 1
                
        elif (meanTapping - epochMean)**2 > (meanControl - epochMean)**2:
            if val > 55:
                confusionMatrix[1,1] += 1
            else:
                confusionMatrix[1,0] += 1
    
    accuracy = (confusionMatrix[0,0] + confusionMatrix[1,1])/np.sum(confusionMatrix)
        
    print(confusionMatrix,accuracy)

    
BaselineClassifier(tapping,control,startTime = 9, stopTime = 11)    