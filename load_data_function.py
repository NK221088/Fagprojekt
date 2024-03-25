from Data_processesing_class import AudioSpeechNoise_data_load, fNIRS_full_motor_data_load, fNIRS_motor_data_load

def load_data(data_set : str, short_channel_correction : bool = None, negative_correlation_enhancement : bool = None):
    if data_set not in ("fNIrs_motor", "AudioSpeechNoise", "fNirs_motor_full_data"):
        raise ValueError("Dataset does not exist.")
    if data_set == "fNIrs_motor":
        all_epochs, data_name, all_data, all_freq, data_types = fNIRS_motor_data_load(short_channel_correction = short_channel_correction, negative_correlation_enhancement = negative_correlation_enhancement).load_data()
        return all_epochs, data_name, all_data, all_freq, data_types
    if data_set == "AudioSpeechNoise":
        all_epochs, data_name, all_data, all_freq, data_types = AudioSpeechNoise_data_load(short_channel_correction = short_channel_correction, negative_correlation_enhancement = negative_correlation_enhancement).load_data()
        return all_epochs, data_name, all_data, all_freq, data_types
    if data_set ==  "fNirs_motor_full_data":
        all_epochs, data_name, all_data, all_freq, data_types = fNIRS_full_motor_data_load(short_channel_correction = short_channel_correction, negative_correlation_enhancement = negative_correlation_enhancement).load_data()
        return all_epochs, data_name, all_data, all_freq, data_types