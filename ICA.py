from load_data_function import load_data
from sklearn.decomposition import FastICA
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler

def ICA(data, keys):
    data = np.vstack((np.median(data[keys[0]], axis = 2), np.median(data[keys[1]], axis = 2)))
    
    scalar = StandardScaler()
    data_standard = scalar.fit_transform(data)
    
    ica = FastICA(n_components=2)
    ica.fit(data_standard)
    
    return ica
    
    
    
    
    