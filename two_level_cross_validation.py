from model import *
from stratified_cv import StratifiedCV


def two_level_cross_validation(modelList, K2, dataset, startTime, stopTime, freq = 7.81):

    
    for participant, data in dataset.items():
        D_test = {data_type: data[data_type] for data_type in data}
        for patient, pdata in dataset.items():
            if patient != participant:
                D_par = {data_type: pdata[data_type] for data_type in pdata}
                
        #Xtrain = np.array(Xtrain).reshape(-1,Xtrain[0].shape[0],Xtrain[0].shape[1])
        tappingArray = D_par["Tapping"]
        controlArray = D_par["Control"]
        E_val = StratifiedCV(modelList = modelList, tappingArray = tappingArray, controlArray = controlArray, startTime = startTime, stopTime = stopTime, freq = freq, K = K2)
        
        E_gen = {}
        
        for model in modelList:
            for param in model.theta:
                liste = []
                for i in range(K2):
                    liste.append((E_val[model.name, i, param][0] * E_val[model.name, i, param][1]) / (len(tappingArray) + len(controlArray)))
                
                E_gen[(model.name, param)] = sum(liste)
                
        print("Break")
                
            
                