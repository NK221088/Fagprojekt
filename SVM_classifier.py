from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

def SVM_classifier(Xtrain, ytrain, Xtest, ytest):
    
    clf = svm.SVC(kernel='rbf')

    # Flatten the last two dimensions of the data
    
    # Standardize the data
    #scaler = StandardScaler()
    #X_standardized = scaler.fit_transform(Xtrain.reshape(Xtrain.shape[0], -1))
    

    
    mu = np.mean(Xtrain)
    std = np.std(Xtrain)
    
    X_standardized = (Xtrain.reshape(Xtrain.shape[0], -1) - mu) / std

    # Apply PCA on the standardized data
    pca = PCA(n_components=2)  # Project data onto the first two principal components
    X_pca = pca.fit_transform(X_standardized)
    
    # Train the SVM classifier on the PCA-transformed and standardized data
    clf.fit(X_pca, ytrain)
    
    
    #X_test_standardized = scaler.transform(Xtest.reshape(Xtest.shape[0], -1))
    
    X_test_standardized = (Xtest.reshape(Xtest.shape[0], -1) - mu) / std
    X_test_pca = pca.transform(X_test_standardized)

    accuracy = clf.score(X = X_test_pca ,y = ytest)

    return accuracy