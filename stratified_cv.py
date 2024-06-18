import numpy as np
from majority_voting_classifier import BaselineModel
from mean_model_classifier import MeanModel
from positive_negative_classifer import Positive_Negative_classifier
from SVM_classifier import SVM_classifier
from ANN import ANN_classifier
from model import model
from tqdm import tqdm
import itertools
from bayes_opt import BayesianOptimization
from ICA import ICA


def StratifiedCV(modelList, tappingArray, controlArray, startTime, stopTime, bayes_opt,use_ica, K = 4, freq = 7.81, **kwargs):
    
    E_val = {}
    
    if bayes_opt:
        for model in modelList:
            E_val[model.name] = {}

    dimTappingArray = tappingArray.shape[0] #Amount of tapping epochs 
    dimControlArray = controlArray.shape[0] #Amount of control epochs
    
    jointArray = np.concatenate((tappingArray,controlArray),axis = 0) #Tapping and control array stacked along the epoch-axis
    jointArray = jointArray[:,:,int(np.floor(startTime * freq)):int(np.floor(stopTime * freq))] #Remove data which are not in specified time interval
    
    randIndTapping = np.random.choice(dimTappingArray, dimTappingArray,replace=False) #Generating list of random indecies for tapping data (0, dimTappingArray)
    randIndControl = np.random.choice(dimControlArray, dimControlArray,replace=False) #Generating list of random indecies for control data (0, dimControlArray)
    randIndControl += dimTappingArray #Shifting random indecies to acount for jointArray (dimTappingArray, dimControlArray + dimTappingArray)
   
    
    
    kernelTapping = int(np.floor(dimTappingArray/K)) #Specifing length of kernel tapping data
    kernelControl = int(np.floor(dimControlArray/K)) #Specifing length of kernel for control data
    
    k0_tapping = 0 #First index of tapping kernel (is updated after each iteration in loop)
    k1_tapping = kernelTapping #Last index of tapping kernel (is updated after each iteration in loop)
    
    k0_control = 0 #First index of control kernel (is updated after each iteration in loop)
    k1_control = kernelControl #Last index of tapping kernel (is updated after each iteration in loop)
    
    with tqdm(total = K, desc = "Inner loop", leave = False, position= 1, ncols = 80) as inner_pbar:
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
            
            train_rand_ind = np.random.choice(size = train_len, a = train_len, replace = False) # Generate random indices for the training data
            test_rand_ind = np.random.choice(size = test_len, a = test_len, replace = False) # Generate random indices for the test data
            
            Xtrain = jointArray[np.concatenate((kernelTappingTrain, kernelControlTrain))[train_rand_ind]]                                           #Extracting training data using indices
            ytrain = np.concatenate((np.ones(len(kernelTappingTrain), dtype = bool), np.zeros(len(kernelControlTrain), dtype = bool)))[train_rand_ind]
            
            Xtest = jointArray[np.concatenate((kernelTappingTest, kernelControlTest))[test_rand_ind]]                                               #Extracting test data using indices
            ytest = np.concatenate((np.ones(len(kernelTappingTest), dtype = bool), np.zeros(len(kernelControlTest), dtype = bool)))[test_rand_ind]
            
            if use_ica == True:
                for model in modelList:
                    model.useICA = True
                
                ica = kwargs['ica']
                
                Xtrain = ica.transform(np.median(Xtrain, axis = 2))
                Xtest = ica.transform(np.median(Xtest, axis = 2))

            if bayes_opt:
                
                for model in modelList:

                    model.load(Xtest = Xtest, Xtrain = Xtrain, ytrain = ytrain, ytest = ytest, n = 2)
                    pbounds = {**model.gaussian_bound,**{f'Feature_{i}': (0,1) for i in range(2)}}   
                    optimizer = BayesianOptimization(f = model.objective_function, pbounds = pbounds)
                    optimizer.maximize(init_points=0,n_iter=10)
                    E_val[model.name][i] = (optimizer.max['target'],optimizer.max['params'])
                    
            else:
                
                for model in modelList:
                    if use_ica:
                        model.theta = {**model.theta,**{f'Feature_{i}': [0,1] for i in range(2)}}
                          
                    param_keys = list(model.theta.keys())
                    param_values = [model.theta[key] for key in param_keys]
                    for combination in itertools.product(*param_values):
                        theta = dict(zip(param_keys, combination))
                        inner_pbar.set_description(f'Currently evaluating ' + model.name + f' on parameter ' + str(theta))
                        model.load(Xtrain = Xtrain, Xtest = Xtest, ytrain = ytrain, ytest = ytest)
                        if use_ica:
                            E_val[(model.name, i, frozenset(theta.items()))] = (model.objective_function(bayes = False, **theta), len(ytest))
                        else:
                            E_val[(model.name, i, frozenset(theta.items()))] = (model.train(theta), len(ytest))
                            
            

            k0_tapping += kernelTapping #Updating kernel.
            k1_tapping += kernelTapping
            k0_control += kernelControl
            k1_control += kernelControl
            inner_pbar.update(1)
    
    return E_val