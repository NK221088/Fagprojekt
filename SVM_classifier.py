from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from seed import set_seeds
set_seeds()

def SVM_classifier(Xtrain, ytrain, Xtest, ytest, theta):
    clf = svm.SVC(kernel=theta["kernel"], C=theta["C"], gamma=theta["gamma"], degree=theta["degree"], coef0=theta["coef0"])
    
    Xtrain = Xtrain.reshape(Xtrain.shape[0], -1)
    Xtest = Xtest.reshape(Xtest.shape[0], -1)
    
    clf.fit(Xtrain, ytrain)
    
    # Predict the test set
    ypred = clf.predict(Xtest)
    
    # Calculate accuracy
    accuracy = clf.score(X=Xtest, y=ytest)
    
    # Compute confusion matrix
    cm = confusion_matrix(ytest, ypred)
    
    return accuracy, cm, (ypred, ytest)



# Flatten the last two dimensions of the data
    
    # Standardize the data
    #scaler = StandardScaler()
    #X_standardized = scaler.fit_transform(Xtrain.reshape(Xtrain.shape[0], -1))
    

    
    # mu = np.mean(Xtrain)
    # std = np.std(Xtrain)
    
    # X_standardized = (Xtrain.reshape(Xtrain.shape[0], -1) - mu) / std

    # Apply PCA on the standardized data
    # pca = PCA(n_components=2)  # Project data onto the first two principal components
    # X_pca = pca.fit_transform(X_standardized)
    
    # Train the SVM classifier on the PCA-transformed and standardized data
    #X_test_standardized = scaler.transform(Xtest.reshape(Xtest.shape[0], -1))
    
    # X_test_standardized = (Xtest.reshape(Xtest.shape[0], -1) - mu) / std
    # X_test_pca = pca.transform(X_test_standardized)