import numpy as np
from majority_voting_classifier import BaselineModel
from mean_model_classifier import MeanModel
from positive_negative_classifer import Positive_Negativ_classifier
from SVM_classifier import SVM_classifier
from ANN import ANN_classifier
from load_data_function import load_data

def individualKFold(individual_data, epoch_type, startTime, stopTime, freq = 7.81):
    
    baselineAccuracy_list = [] #List to store accuracies
    meanModelAccuracy_list = []
    psAccuracy_list = []
    svm_accuracy_list = []
    ANN_accuracy_list = []

    for participant, data in individual_data.items():
        if len(data[epoch_type]) > 0 and len(data["Control"]) > 0:
            Xtest = np.concatenate([data[data_type] for data_type in [epoch_type, "Control"]])
            ytest = np.concatenate([np.ones(len(data[epoch_type]), dtype = bool), np.zeros(len(data["Control"]), dtype = bool)])
            Xtrain = []
            ytrain = []
            for patient, pdata in individual_data.items():
                if patient != participant:
                    Xtrain.extend(np.concatenate([pdata[data_type] for data_type in [epoch_type, "Control"]]))
                    ytrain.extend(np.concatenate([np.ones(len(pdata[epoch_type]), dtype = bool), np.zeros(len(pdata["Control"]), dtype = bool)]))
            Xtrain = np.array(Xtrain).reshape(-1,Xtrain[0].shape[0],Xtrain[0].shape[1])
            ytrain = np.array(ytrain).flatten()
            
            # Generate shuffled indices
            train_indices = np.random.choice(len(Xtrain), len(Xtrain), replace=False)
            test_indices = np.random.choice(len(Xtest), len(Xtest), replace=False)

            # Shuffle Xtrain and ytrain using the same set of indices
            Xtrain = Xtrain[train_indices]
            ytrain = ytrain[train_indices]
            
            Xtest = Xtest[test_indices]
            ytest = ytest[test_indices]
            
            meanModel_accuracy = MeanModel(Xtrain = Xtrain,  ytrain = ytrain, Xtest = Xtest, ytest = ytest)
            baselineaccuracy = BaselineModel(Xtrain = Xtrain,  ytrain = ytrain, Xtest = Xtest, ytest = ytest)
            ps_accuracy = Positive_Negativ_classifier(Xtrain = Xtrain,  ytrain = ytrain, Xtest = Xtest, ytest = ytest)
            svm_accuracy = SVM_classifier(Xtrain = Xtrain,  ytrain = ytrain, Xtest = Xtest, ytest = ytest)
            ANN_error, ANN_accuracy = ANN_classifier(Xtrain = Xtrain,  ytrain = ytrain, Xtest = Xtest, ytest = ytest)        

            meanModelAccuracy_list.append(meanModel_accuracy)
            baselineAccuracy_list.append(baselineaccuracy)
            psAccuracy_list.append(ps_accuracy)
            svm_accuracy_list.append(svm_accuracy)
            ANN_accuracy_list.append(ANN_accuracy)
    
    return {"MeanModel": meanModelAccuracy_list, "MajorityVoting": baselineAccuracy_list, "PSModel": psAccuracy_list, "SVMModel": svm_accuracy_list, "ANNModel": ANN_accuracy_list}