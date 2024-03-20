import numpy as np
from majority_voting_classifier import BaselineModel
from mean_model_classifier import MeanModel

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

        meanModel_accuracy,_ = MeanModel(TappingTest = kernelTappingTest, ControlTest= kernelControlTest, TappingTrain = kernelTappingTrain, ControlTrain= kernelControlTrain, jointArray=jointArray, labelIndx = tappingArray.shape[0])
        baseline_accuracy = BaselineModel(TappingTest = kernelTappingTest, ControlTest= kernelControlTest, TappingTrain = kernelTappingTrain, ControlTrain= kernelControlTrain)
        
        meanModelAccuracy_list.append(meanModel_accuracy)
        baselineAccuracy_list.append(baseline_accuracy)
    
        
        k0_tapping += kernelTapping #Updating kernel.
        k1_tapping += kernelTapping
        
        k0_control += kernelControl
        k1_control += kernelControl
        
    return str(np.round(np.mean(meanModelAccuracy_list), 3)) + u"\u00B1" + str(np.round(1.96 * np.std(meanModelAccuracy_list)/np.sqrt(len(meanModelAccuracy_list)),3)), str(np.round(np.mean(baselineAccuracy_list),3)) + u"\u00B1" + str(np.round(1.96 * np.std(baselineAccuracy_list)/np.sqrt(len(baselineAccuracy_list)),3))