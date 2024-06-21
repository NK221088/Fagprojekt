from seed import set_seeds
import numpy as np

set_seeds()
class individual_participant_class:
    def __init__(self, name):
        self.name = name
        self.events = {}
        self.raw_intensity = None
        self.raw_od = None
        self.raw_haemo_unfiltered = None
        self.raw_haemo = None
    
    def Standardize(self):
        for datatype, data in self.events.items():
    # Compute the mean for each channel over epochs and time
            channel_mean = np.mean(data, axis=(0, 2), keepdims=True)
    
    # Compute the standard deviation over the entire dataset
            global_std = np.std(data, axis=(0, 1, 2), keepdims=True)
    
    # Subtract the channel mean and divide by the global standard deviation
            normalized_data = (data - channel_mean) / global_std
    
    # Store the normalized data back into the dictionary
            self.events[datatype] = normalized_data