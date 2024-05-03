import numpy as np
def BaselineModel(Xtrain, ytrain, Xtest, ytest, theta):
    N = len(ytest)    
    if sum(ytrain == 1) < sum(ytrain == 0):
        return sum(ytest == 0) / N
    else:
        return sum(ytest == 1) / N