from model import *
from stratified_cv import StratifiedCV


def two_level_cross_validation(modelList, K2, dataset, startTime, stopTime, freq = 7.81):

    N = sum([data[data_type].shape[0] for participant, data in dataset.items() for data_type in data])
    E_test = {model.name: {} for model in modelList}
    count = 0
    
    
    for participant, data in dataset.items():
        D_test = {data_type: data[data_type] for data_type in data}
        tappingArray_par = np.array([])
        controlArray_par = np.array([])
        D_par = {'Tapping': [], 'Control': []}
        for patient, pdata in dataset.items():
            if patient != participant:
                for data_type in pdata:
                    D_par[data_type].append(pdata[data_type])
        
        for data_type in data:
            D_par[data_type] = np.vstack(D_par[data_type])
                
        
        tappingArray_par = D_par["Tapping"]
        controlArray_par = D_par["Control"]
        
        tappingArray_test = D_test["Tapping"]
        controlArray_test = D_test["Control"]
        
        
        E_val = StratifiedCV(modelList = modelList, tappingArray = tappingArray_par, controlArray = controlArray_par, startTime = startTime, stopTime = stopTime, freq = freq, K = K2)
        
        E_gen = {}  # Initialize the outer dictionary

        for model in modelList:
            E_gen[model.name] = {}  # Initialize a new dictionary for each model inside E_gen
            for theta in model.theta:
                liste = []
                for i in range(K2):
                    # Calculate and append values to the list for each i
                    liste.append((E_val[model.name, i, theta][0] * E_val[model.name, i, theta][1]) / 
                                (len(tappingArray_par) + len(controlArray_par)))
                
                # Sum the list and assign it to the inner dictionary under the key theta
                E_gen[model.name][theta] = sum(liste)


        theta_star = [max(E_gen[model.name], key=E_gen[model.name].get) for model in modelList]
        
        train_size = tappingArray_par.shape[0] + controlArray_par.shape[0]
        test_size =  tappingArray_test.shape[0] + controlArray_test.shape[0]           
        
        train_randomizer = np.random.choice(size = train_size, a = train_size, replace = False)
        test_randomizer = np.random.choice(size = test_size, a = test_size, replace=False)
        
        train_set = np.concatenate((tappingArray_par,controlArray_par), axis = 0)[train_randomizer]
        test_set = np.concatenate((tappingArray_test, controlArray_test), axis = 0)[test_randomizer]
        
        ytrain = np.concatenate((np.ones(tappingArray_par.shape[0]), np.zeros(controlArray_par.shape[0])))[train_randomizer]
        ytest = np.concatenate((np.ones(tappingArray_test.shape[0]), np.zeros(controlArray_test.shape[0])))[test_randomizer] 
        
        for i, model in enumerate(modelList):
            E_test[model.name][count] = (model.train(Xtrain = train_set, ytrain = ytrain, Xtest = test_set, ytest = ytest, theta = theta_star[i]), test_size)
        
        count += 1
    
    E_gen_hat = {model.name: 0 for model in modelList}
    
    for model in E_test.keys():
        for i in range(5):
            E_gen_hat[model] += E_test[model][i][0] * (E_test[model][i][1] / N)
            
    return E_gen_hat
                
                    
                    
                    
                
                    