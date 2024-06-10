from Data_processesing_class import AudioSpeechNoise_data_load, fNIRS_full_motor_data_load, fNIRS_motor_data_load, fNIRS_Alexandros_DoC_data_load, fNIRS_Alexandros_Healthy_data_load

def load_data(data_set : str, short_channel_correction : bool = None, negative_correlation_enhancement : bool = None, individuals :bool = False, interpolate_bad_channels:bool=False):
    if data_set not in ("fNIrs_motor", "AudioSpeechNoise", "fNirs_motor_full_data", "fNIRS_Alexandros_DoC_data", "fNIRS_Alexandros_Healthy_data"):
        raise ValueError("Dataset does not exist.")
    if data_set == "fNIrs_motor":
        if individuals:
            all_epochs, data_name, all_data, all_freq, data_types, individual_data = fNIRS_motor_data_load(short_channel_correction = short_channel_correction, negative_correlation_enhancement = negative_correlation_enhancement, individuals = individuals, interpolate_bad_channels=interpolate_bad_channels).load_data()
            return all_epochs, data_name, all_data, all_freq, data_types, individual_data
        else:
            all_epochs, data_name, all_data, all_freq, data_types = fNIRS_motor_data_load(short_channel_correction = short_channel_correction, negative_correlation_enhancement = negative_correlation_enhancement, individuals = individuals, interpolate_bad_channels=interpolate_bad_channels).load_data()
            return all_epochs, data_name, all_data, all_freq, data_types
    if data_set == "AudioSpeechNoise":
        if individuals:
            all_epochs, data_name, all_data, all_freq, data_types, individual_data = AudioSpeechNoise_data_load(short_channel_correction = short_channel_correction, negative_correlation_enhancement = negative_correlation_enhancement, individuals = individuals, interpolate_bad_channels=interpolate_bad_channels).load_data()
            return all_epochs, data_name, all_data, all_freq, data_types, individual_data
        else:
            all_epochs, data_name, all_data, all_freq, data_types = AudioSpeechNoise_data_load(short_channel_correction = short_channel_correction, negative_correlation_enhancement = negative_correlation_enhancement, individuals = individuals, interpolate_bad_channels=interpolate_bad_channels).load_data()
            return all_epochs, data_name, all_data, all_freq, data_types
    if data_set ==  "fNirs_motor_full_data":
        if individuals:
            all_epochs, data_name, all_data, all_freq, data_types, individual_data = fNIRS_full_motor_data_load(short_channel_correction = short_channel_correction, negative_correlation_enhancement = negative_correlation_enhancement, individuals = individuals, interpolate_bad_channels=interpolate_bad_channels).load_data()
            return all_epochs, data_name, all_data, all_freq, data_types, individual_data
        else:
            all_epochs, data_name, all_data, all_freq, data_types = fNIRS_full_motor_data_load(short_channel_correction = short_channel_correction, negative_correlation_enhancement = negative_correlation_enhancement, individuals = individuals, interpolate_bad_channels=interpolate_bad_channels).load_data()
            return all_epochs, data_name, all_data, all_freq, data_types
    if data_set ==  "fNIRS_Alexandros_DoC_data":
        if individuals:
            all_epochs, data_name, all_data, all_freq, data_types, individual_data = fNIRS_Alexandros_DoC_data_load(short_channel_correction = short_channel_correction, negative_correlation_enhancement = negative_correlation_enhancement, individuals = individuals, interpolate_bad_channels=interpolate_bad_channels).load_data()
            return all_epochs, data_name, all_data, all_freq, data_types, individual_data
        else:
            all_epochs, data_name, all_data, all_freq, data_types = fNIRS_Alexandros_DoC_data_load(short_channel_correction = short_channel_correction, negative_correlation_enhancement = negative_correlation_enhancement, individuals = individuals, interpolate_bad_channels=interpolate_bad_channels).load_data()
            return all_epochs, data_name, all_data, all_freq, data_types
    if data_set ==  "fNIRS_Alexandros_Healthy_data":
        if individuals:
            all_epochs, data_name, all_data, all_freq, data_types, individual_data = fNIRS_Alexandros_Healthy_data_load(short_channel_correction = short_channel_correction, negative_correlation_enhancement = negative_correlation_enhancement, individuals = individuals, interpolate_bad_channels=interpolate_bad_channels).load_data()
            return all_epochs, data_name, all_data, all_freq, data_types, individual_data
        else:
            all_epochs, data_name, all_data, all_freq, data_types = fNIRS_Alexandros_Healthy_data_load(short_channel_correction = short_channel_correction, negative_correlation_enhancement = negative_correlation_enhancement, individuals = individuals, interpolate_bad_channels=interpolate_bad_channels).load_data()
            return all_epochs, data_name, all_data, all_freq, data_types