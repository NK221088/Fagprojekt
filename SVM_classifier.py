from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import tensorflow as tf
import numpy as np
import random
import os

# Set seeds for reproducibility
def set_seeds(seed=42):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # If using TensorFlow with GPU:
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

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
    
    return accuracy, cm



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