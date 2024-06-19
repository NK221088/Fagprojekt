import numpy as np
from majority_voting_classifier import BaselineModel
from mean_model_classifier import MeanModel
from positive_negative_classifer import Positive_Negativ_classifier
from SVM_classifier import SVM_classifier
from ANN1 import ANN_classifier
from load_data_function import load_data

# Data set:
data_set = "fNirs_motor_full_data"
epoch_type = "Tapping"
combine_strategy = "mean"
individuals = True

# Data processing:
bad_channels_strategy = "all"
short_channel_correction = True
negative_correlation_enhancement = True
threshold = 3
startTime = 7.5
stopTime = 12.5
K = 10

all_epochs, data_name, all_data, freq, data_types, all_individuals = load_data(data_set = data_set, short_channel_correction = short_channel_correction, negative_correlation_enhancement = negative_correlation_enhancement, individuals = individuals)

def StratifiedCV(individual_data, startTime, stopTime, freq = 7.81):
    

    
    
    baselineAccuracy_list = [] #List to store accuracies
    meanModelAccuracy_list = []
    psAccuracy_list = []
    svm_accuracy_list = []
    ANN_accuracy_list = []

    # dimTappingArray = tappingArray.shape[0] #Amount of tapping epochs 
    # dimControlArray = controlArray.shape[0] #Amount of control epochs
    
    # jointArray = np.concatenate((tappingArray,controlArray),axis = 0) #Tapping and control array stacked along the epoch-axis
    # jointArray = jointArray[:,:,int(np.floor(startTime * freq)):int(np.floor(stopTime * freq))] #Remove data which are not in specified time interval
    
    # randIndTapping = np.random.choice(dimTappingArray, dimTappingArray,replace=False) #Generating list of random indecies for tapping data (0, dimTappingArray)
    # randIndControl = np.random.choice(dimControlArray, dimControlArray,replace=False) #Generating list of random indecies for control data (0, dimControlArray)
    # randIndControl += dimTappingArray #Shifting random indecies to acount for jointArray (dimTappingArray, dimControlArray + dimTappingArray)
   
    
    
    # kernelTapping = int(np.ceil(dimTappingArray/K)) #Specifing length of kernel tapping data
    # kernelControl = int(np.ceil(dimControlArray/K)) #Specifing length of kernel for control data
    
    # k0_tapping = 0 #First index of tapping kernel (is updated after each iteration in loop)
    # k1_tapping = kernelTapping #Last index of tapping kernel (is updated after each iteration in loop)
    
    # k0_control = 0 #First index of control kernel (is updated after each iteration in loop)
    # k1_control = kernelControl #Last index of tapping kernel (is updated after each iteration in loop)
    
    # for i in range(K):
        
        # if k1_tapping > dimTappingArray: #Cutting last index of tapping kernel if too long
        #     k1_tapping = dimTappingArray 
        
        # if k1_control > dimControlArray: #Cutting last index of control kernel if too long
        #     k1_control = dimControlArray
        
        
        # kernelTappingTest = randIndTapping[k0_tapping:k1_tapping] #Selecting tapping kernel indecies (test data)  
        # kernelTappingTrain = np.concatenate((randIndTapping[:k0_tapping],randIndTapping[k1_tapping:])) #Selecting all indecies outside of kernel (train data)
    
        # kernelControlTest = randIndControl[k0_control:k1_control] #Selecting control kernel indecies (test data)  
        # kernelControlTrain = np.concatenate((randIndControl[:k0_control],randIndControl[k1_control:])) #Selecting control kernel indecies (train data)

        # train_len = len(kernelControlTrain) + len(kernelTappingTrain)
        # test_len = len(kernelTappingTest) + len(kernelControlTest)
        
        
        # train_rand_ind = np.random.choice(size = train_len, a = train_len)
        # test_rand_ind = np.random.choice(size = test_len, a = test_len)
        
        # Xtrain = jointArray[np.concatenate((kernelTappingTrain, kernelControlTrain))[train_rand_ind]]
        # ytrain = np.concatenate((np.ones(len(kernelTappingTrain), dtype = bool), np.zeros(len(kernelControlTrain), dtype = bool)))[train_rand_ind]
        
        # Xtest = jointArray[np.concatenate((kernelTappingTest, kernelControlTest))[test_rand_ind]]
        # ytest = np.concatenate((np.ones(len(kernelTappingTest), dtype = bool), np.zeros(len(kernelControlTest), dtype = bool)))[test_rand_ind]


    for participant, data in all_individuals.items():
        Xtest = np.concatenate([data[data_type] for data_type in data])
        ytest = np.concatenate([np.ones(len(data["Tapping"])), np.zeros(len(data["Control"]))])
        Xtrain = []
        ytrain = []
        for patient, pdata in all_individuals.items():
            if patient != participant:
                Xtrain.extend(np.concatenate([pdata[data_type] for data_type in pdata]))
                ytrain.extend(np.concatenate([np.ones(len(pdata["Tapping"])), np.zeros(len(pdata["Control"]))]))
        Xtrain = np.array(Xtrain).reshape(-1,Xtrain[0].shape[0],Xtrain[0].shape[1])
        ytrain = np.array(ytrain).flatten()
        
        # meanModel_accuracy = MeanModel(Xtrain = Xtrain,  ytrain = ytrain, Xtest = Xtest, ytest = ytest)
        # baselineaccuracy = BaselineModel(TappingTest = kernelTappingTest, ControlTest= kernelControlTest, TappingTrain = kernelTappingTrain, ControlTrain= kernelControlTrain)
        # ps_accuracy = Positive_Negativ_classifier(TappingTest = kernelTappingTest, ControlTest= kernelControlTest, TappingTrain = kernelTappingTrain, ControlTrain= kernelControlTrain,jointArray=jointArray, labelIndx = tappingArray.shape[0])
        svm_accuracy = SVM_classifier(Xtrain = Xtrain,  ytrain = ytrain, Xtest = Xtest, ytest = ytest)
        ANN_error, ANN_accuracy = ANN_classifier(Xtrain = Xtrain,  ytrain = ytrain, Xtest = Xtest, ytest = ytest)        

        # meanModelAccuracy_list.append(meanModel_accuracy)
        # baselineAccuracy_list.append(baselineaccuracy)
        # psAccuracy_list.append(ps_accuracy)
        svm_accuracy_list.append(svm_accuracy)
        ANN_accuracy_list.append(ANN_accuracy)
        
        # k0_tapping += kernelTapping #Updating kernel.
        # k1_tapping += kernelTapping
        
        # k0_control += kernelControl
        # k1_control += kernelControl
    
    return {"MeanModel": meanModelAccuracy_list, "MajorityVoting": baselineAccuracy_list, "PSModel": psAccuracy_list, "SVMModel": svm_accuracy_list, "ANNModel": ANN_accuracy_list}

StratifiedCV(individual_data = all_individuals, startTime = 7.5, stopTime = 12.5)