from load_data_function import load_data
from sklearn.decomposition import FastICA
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler

def ICA(data):
    
    
    