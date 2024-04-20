import numpy as np
from majority_voting_classifier import BaselineModel
from mean_model_classifier import MeanModel
from positive_negative_classifer import Positive_Negativ_classifier
from SVM_classifier import SVM_classifier
from ANN import ANN_classifier

def StratifiedCV(tappingArray, controlArray, startTime, stopTime, K = 4, freq = 7.81):
    
    baselineAccuracy_list = [] #List to store accuracies
    meanModelAccuracy_list = []
    psAccuracy_list = []
    svm_accuracy_list = []
    ANN_accuracy_list = []

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

        train_len = len(kernelControlTrain) + len(kernelTappingTrain)
        test_len = len(kernelTappingTest) + len(kernelControlTest)
        
        
        train_rand_ind = np.random.choice(size = train_len, a = train_len)
        test_rand_ind = np.random.choice(size = test_len, a = test_len)
        
        Xtrain = jointArray[np.concatenate((kernelTappingTrain, kernelControlTrain))[train_rand_ind]]
        ytrain = np.concatenate((np.ones(len(kernelTappingTrain), dtype = bool), np.zeros(len(kernelControlTrain), dtype = bool)))[train_rand_ind]
        
        Xtest = jointArray[np.concatenate((kernelTappingTest, kernelControlTest))[test_rand_ind]]
        ytest = np.concatenate((np.ones(len(kernelTappingTest), dtype = bool), np.zeros(len(kernelControlTest), dtype = bool)))[test_rand_ind]
        
        
        meanModel_accuracy = MeanModel(Xtrain = Xtrain,  ytrain = ytrain, Xtest = Xtest, ytest = ytest)
        baselineaccuracy = BaselineModel(Xtrain = Xtrain,  ytrain = ytrain, Xtest = Xtest, ytest = ytest)
        ps_accuracy = Positive_Negativ_classifier(TappingTest = kernelTappingTest, ControlTest= kernelControlTest, TappingTrain = kernelTappingTrain, ControlTrain= kernelControlTrain,jointArray=jointArray, labelIndx = tappingArray.shape[0])
        svm_accuracy = SVM_classifier(Xtrain = Xtrain,  ytrain = ytrain, Xtest = Xtest, ytest = ytest)
        ANN_error, ANN_accuracy = ANN_classifier(Xtrain = Xtrain,  ytrain = ytrain, Xtest = Xtest, ytest = ytest)        

        meanModelAccuracy_list.append(meanModel_accuracy)
        baselineAccuracy_list.append(baselineaccuracy)
        psAccuracy_list.append(ps_accuracy)
        svm_accuracy_list.append(svm_accuracy)
        ANN_accuracy_list.append(ANN_accuracy)
        
        k0_tapping += kernelTapping #Updating kernel.
        k1_tapping += kernelTapping
        
        k0_control += kernelControl
        k1_control += kernelControl
    
    return {"MeanModel": meanModelAccuracy_list, "MajorityVoting": baselineAccuracy_list, "PSModel": psAccuracy_list, "SVMModel": svm_accuracy_list, "ANNModel": ANN_accuracy_list}