from fNirs_processesing_fNirs_motor import data_fNirs_motor
from fnirs_processing_AudioSpeechNoise import all_epochs, epochs, all_data, all_freq, data_name
from fnirs_processing_AudioSpeechNoise_SCC import data_AudioSpeechNoise
from fnirs_processing_fnirs_motor_full_data import data_fNirs_motor_full_data

def load_data(data_set : str, short_channel_correction : bool = None, negative_correlation_enhancement : bool = None):
    if data_set == "fNIrs_motor":
        all_epochs, data_name, all_data, all_freq = data_fNirs_motor()
        return all_epochs, data_name, all_data, all_freq
    if data_set == "AudioSpeechNoise":
        all_epochs, data_name, all_data, all_freq = data_AudioSpeechNoise(short_channel_correction, negative_correlation_enhancement)
        return all_epochs, data_name, all_data, all_freq
    if data_set ==  "fNirs_motor_full_data":
        all_epochs, data_name, all_data, all_freq = data_fNirs_motor_full_data()
        return all_epochs, data_name, all_data, all_freq