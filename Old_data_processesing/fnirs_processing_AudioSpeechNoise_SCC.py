"""
.. _tut-fnirs-processing:

================================================================
Preprocessing functional near-infrared spectroscopy (fNIRS) data
================================================================

This tutorial covers how to convert functional near-infrared spectroscopy
(fNIRS) data from raw measurements to relative oxyhaemoglobin (HbO) and
deoxyhaemoglobin (HbR) concentration, view the average waveform, and
topographic representation of the response.

Here we will work with the :ref:`fNIRS motor data <fnirs-motor-dataset>`.
"""
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.
# %%

from itertools import compress
import matplotlib.pyplot as plt
import numpy as np

import mne
import mne_nirs
import mne_bids
import os

def data_AudioSpeechNoise(short_channel_correction : bool, negative_correlation_enhancement : bool):
    all_epochs = []
    all_speech = []
    all_noise = []
    all_control = []

    fnirs_snirf_file = mne_nirs.datasets.block_speech_noise.data_path()
    # Loop over all subjects
    for i in range(1, 18):
        sub_id = str(i).zfill(2)  # Pad with zeros to get "01", "02", etc.
        fnirs_snirf_file_path = os.path.join(fnirs_snirf_file, f"sub-{sub_id}", "ses-01", "nirs", f"sub-{sub_id}_ses-01_task-AudioSpeechNoise_nirs.snirf")
        raw_intensity = mne.io.read_raw_snirf(fnirs_snirf_file_path, verbose=True)
        raw_intensity.load_data()

        raw_intensity.annotations.set_durations(5)
        raw_intensity.annotations.rename(
            {"1.0": "Control", "2.0": "Activity/Noise", "3.0": "Activity/Speech"}
        )
        unwanted = np.nonzero(raw_intensity.annotations.description == "15.0")
        raw_intensity.annotations.delete(unwanted)
        

        if short_channel_correction:
            raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)
            raw_od = mne_nirs.signal_enhancement.short_channel_regression(raw_od)
            
            picks = mne.pick_types(raw_intensity.info, meg=False, fnirs=True)
            dists = mne.preprocessing.nirs.source_detector_distances(
                raw_intensity.info, picks=picks
            )
            raw_intensity.pick(picks[dists > 0.01])
        else:
            picks = mne.pick_types(raw_intensity.info, meg=False, fnirs=True)
            dists = mne.preprocessing.nirs.source_detector_distances(
                raw_intensity.info, picks=picks
            )
            raw_intensity.pick(picks[dists > 0.01])
            raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)

        sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od)



        raw_od.info["bads"] = list(compress(raw_od.ch_names, sci < 0.5))
        
        raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=0.1)
            
        raw_haemo_unfiltered = raw_haemo.copy()
        raw_haemo.filter(0.05, 0.7, h_trans_bandwidth=0.2, l_trans_bandwidth=0.02)
        
        if negative_correlation_enhancement:
            raw_haemo = mne_nirs.signal_enhancement.enhance_negative_correlation(raw_haemo)


        events, event_dict = mne.events_from_annotations(raw_haemo)
        


        reject_criteria = dict(hbo=80e-6)
        tmin, tmax = -5, 15

        epochs = mne.Epochs(
            raw_haemo,
            events,
            event_id=event_dict,
            tmin=tmin,
            tmax=tmax,
            reject=reject_criteria,
            reject_by_annotation=True,
            proj=True,
            baseline=(None, 0),
            preload=True,
            detrend=None,
            verbose=True,
        )
        
        all_epochs.append(epochs)
        all_control.append(epochs["Control"].get_data())
        all_speech.append(epochs["Speech"].get_data())
        all_noise.append(epochs["Noise"].get_data())

    all_control = np.concatenate(all_control, axis = 0)
    all_speech = np.concatenate(all_speech, axis = 0)
    all_noise = np.concatenate(all_noise, axis = 0)

    data_name = "AudioSpeechNoise"
    all_data = {"Control": all_control, "Speech": all_speech, "Noise": all_noise}
    all_freq = all_epochs[0].info['sfreq']
    return all_epochs, data_name, all_data, all_freq