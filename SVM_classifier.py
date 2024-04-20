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
    
        # Create a mesh grid based on the range of PCA transformed features
    h = .2  # step size in the mesh
    x_min, x_max = X_test_pca[:, 0].min() - 1, X_test_pca[:, 0].max() + 1
    y_min, y_max = X_test_pca[:, 1].min() - 1, X_test_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict on the mesh grid
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=ytest, edgecolors='k', cmap=plt.cm.Paired)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Decision surface using PCA-transformed/projected features')
    plt.show()

    return accuracy