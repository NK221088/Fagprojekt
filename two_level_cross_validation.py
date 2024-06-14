from model import *
from stratified_cv import StratifiedCV
from tqdm import tqdm
import itertools
from ICA import ICA

def two_level_cross_validation(modelList, K2, dataset, startTime, stopTime, bayes_opt, freq = 7.81, use_ica = True):
    
    

    dataset = {participant.name: participant.events for participant in dataset}
    
    
    dataset_array = {'Tapping': [], 'Control': []}
    
    for patient, pdata in dataset.items():
              
        for data_type in pdata:
            dataset_array[data_type].append(pdata[data_type])
    
    
    
    E_genList = []

    N = sum([data[data_type].shape[0] for participant, data in dataset.items() for data_type in data])
    E_test = {model.name: {} for model in modelList}
    count = 0

    theta_list = []
    
    with tqdm(total = len(dataset), desc="Outer loop", position=0, ncols = 100) as outer_pbar:
        for participant, data in dataset.items():
            outer_pbar.set_description(f'{participant}')

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
            
            train_size = tappingArray_par.shape[0] + controlArray_par.shape[0]
            test_size =  tappingArray_test.shape[0] + controlArray_test.shape[0]           
            
            train_randomizer = np.random.choice(size = train_size, a = train_size, replace = False)
            test_randomizer = np.random.choice(size = test_size, a = test_size, replace=False)
            
            train_set = np.concatenate((tappingArray_par,controlArray_par), axis = 0)[train_randomizer]
            test_set = np.concatenate((tappingArray_test, controlArray_test), axis = 0)[test_randomizer]
            
            ytrain = np.concatenate((np.ones(tappingArray_par.shape[0]), np.zeros(controlArray_par.shape[0])))[train_randomizer]
            ytest = np.concatenate((np.ones(tappingArray_test.shape[0]), np.zeros(controlArray_test.shape[0])))[test_randomizer] 
            
            
            E_val = StratifiedCV(modelList = modelList, tappingArray = tappingArray_par, controlArray = controlArray_par, startTime = startTime, stopTime = stopTime, freq = freq, K = K2, n_features=5, use_ica = use_ica, bayes_opt=bayes_opt)
            
            outer_pbar.update(1)
            E_gen = {}  # Initialize the outer dictionary

            if bayes_opt:
                for model in modelList:
                    theta_star = max(E_val[model.name], key = E_val[model.name].get)
                    
                    Xtrain, Xtest = ICA(Xtrain = train_set, Xtest = test_set, n_components = 5, plot = False, save_plot = False)
                    
                    model.load(Xtrain = Xtrain, Xtest = Xtest, ytrain = ytrain, ytest = ytest, n = 5)
                    
                    
                    E_test[model.name][count] = (model.objective_function(**E_val[model.name][theta_star][1]), test_size)
                    
            else:
                
                for model in modelList:
                    
                    E_gen[model.name] = {}  # Initialize a new dictionary for each model inside E_gen
                    param_keys = list(model.theta.keys())
                    param_values = [model.theta[key] for key in param_keys]
                    
                    for combination in itertools.product(*param_values):
                        theta = dict(zip(param_keys, combination))
                        values = []
                        
                        for i in range(K2):
                            # Calculate and append values to the list for each i
                            values.append((E_val[model.name, i, frozenset(theta.items())][0] * E_val[model.name, i, frozenset(theta.items())][1]) / 
                                        (len(tappingArray_par) + len(controlArray_par)))
                        
                        # Sum the list and assign it to the inner dictionary under the key theta
                        E_gen[model.name][frozenset(theta.items())] = sum(values)

                E_genList.append(E_gen)
                theta_star = [max(E_gen[model.name], key=E_gen[model.name].get) for model in modelList]                
                for i, model in enumerate(modelList):
                    E_test[model.name][count] = (model.train(Xtrain = train_set, ytrain = ytrain, Xtest = test_set, ytest = ytest, theta = dict(theta_star[i])), test_size)
            
            count += 1
    
    E_gen_hat = {model.name: 0 for model in modelList}
    
    for model in E_test.keys():
        for i in range(len(dataset)):
            E_gen_hat[model] += E_test[model][i][0] * (E_test[model][i][1] / N)
            
    return E_gen_hat,E_genList
                
                    
                    
                    
                
                    