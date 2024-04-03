from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

def SVM_classifier(TappingTest, ControlTest, TappingTrain, ControlTrain, jointArray, labelIndx):
    confusionMatrix = np.zeros((2, 2))  # 0 = control, 1 = Tapping
    clf = svm.SVC(kernel='rbf')

    # Flatten the last two dimensions of the data
    train_indices = np.concatenate((TappingTrain, ControlTrain))
    X = jointArray[train_indices].reshape(len(train_indices), -1)
    
    # Standardize the data
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)

    # Apply PCA on the standardized data
    pca = PCA(n_components=2)  # Project data onto the first two principal components
    X_pca = pca.fit_transform(X_standardized)

    y = np.concatenate((np.ones(len(TappingTrain)), np.zeros(len(ControlTrain))))
    
    # Train the SVM classifier on the PCA-transformed and standardized data
    clf.fit(X_pca, y)

    for val in np.concatenate((TappingTest, ControlTest), axis=0):
        # Standardize and project the test data onto the same PCA components
        X_test = jointArray[val].reshape(1, -1)
        X_test_standardized = scaler.transform(X_test)  # Use transform, not fit_transform
        X_test_pca = pca.transform(X_test_standardized)

        pred = clf.predict(X_test_pca)[0]

        if val < labelIndx:
            if pred == 1:
                confusionMatrix[0, 0] += 1
            else:
                confusionMatrix[0, 1] += 1
        else:
            if pred == 0:
                confusionMatrix[1, 1] += 1
            else:
                confusionMatrix[1, 0] += 1

    accuracy = (confusionMatrix[0, 0] + confusionMatrix[1, 1]) / np.sum(confusionMatrix)

    return accuracy, confusionMatrix
