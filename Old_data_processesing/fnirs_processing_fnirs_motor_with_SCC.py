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
from _short_channel_correction import short_channel_regression



all_epochs_anti = []
all_tapping_anti = []
all_control_anti = []

fnirs_snirf_file = mne_nirs.datasets.block_speech_noise.data_path()
# Loop over all subjects
for i in range(1, 2):
    
    fnirs_data_folder = mne.datasets.fnirs_motor.data_path()
    fnirs_cw_amplitude_dir = fnirs_data_folder / "Participant-1"
    raw_intensity = mne.io.read_raw_nirx(fnirs_cw_amplitude_dir, verbose=True)
    raw_intensity.load_data()

    raw_intensity.annotations.set_durations(5)
    raw_intensity.annotations.rename(
                                    {"1.0": "Control",
                                    "2.0": "Tapping/Left",
                                    "3.0": "Tapping/Right"})
    unwanted = np.nonzero(raw_intensity.annotations.description == "15.0")
    raw_intensity.annotations.delete(unwanted)


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
    raw_haemo = raw_haemo.filter(0.05, 0.7, h_trans_bandwidth=0.2, l_trans_bandwidth=0.02)

    events, event_dict = mne.events_from_annotations(raw_haemo)

    reject_criteria = dict(hbo=80e-6)
    tmin, tmax = -5, 15

    raw_anti = mne_nirs.signal_enhancement.enhance_negative_correlation(raw_haemo)

    epochs_anti = mne.Epochs(raw_anti, events, event_id=event_dict,
                            tmin=tmin, tmax=tmax,
                            reject=reject_criteria, reject_by_annotation=True,
                            proj=True, baseline=(None, 0), preload=True,
                            detrend=None, verbose=True)

    evoked_dict_anti = {'Tapping/HbO': epochs_anti['Tapping'].average(picks='hbo'),
                        'Tapping/HbR': epochs_anti['Tapping'].average(picks='hbr'),
                        'Control/HbO': epochs_anti['Control'].average(picks='hbo'),
                        'Control/HbR': epochs_anti['Control'].average(picks='hbr')}
    
    all_epochs_anti.append(epochs_anti)
    all_tapping_anti.append(epochs_anti["Tapping"].get_data())
    all_control_anti.append(epochs_anti["Control"].get_data())
    
all_tapping_anti = np.concatenate(all_tapping_anti, axis = 0)
all_control_anti = np.concatenate(all_control_anti, axis = 0)

data_name_anti = "fnirs_motor_plus_anti"
all_data_anti = {"Tapping": all_tapping_anti, "Control": all_control_anti}
all_freq_anti = all_epochs_anti[0].info['sfreq']
