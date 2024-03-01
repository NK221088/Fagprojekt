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

fnirs_data_folder = mne.datasets.fnirs_motor.data_path()
fnirs_cw_amplitude_dir = fnirs_data_folder / "Participant-1"
raw_intensity = mne.io.read_raw_nirx(fnirs_cw_amplitude_dir, verbose=True)
raw_intensity.load_data()


# %%
# Providing more meaningful annotation information
# ------------------------------------------------
#
# First, we attribute more meaningful names to the trigger codes which are
# stored as annotations. Second, we include information about the duration of
# each stimulus, which was 5 seconds for all conditions in this experiment.
# Third, we remove the trigger code 15, which signaled the start and end
# of the experiment and is not relevant to our analysis.

raw_intensity.annotations.set_durations(5)
raw_intensity.annotations.rename(
    {"1.0": "Control", "2.0": "Tapping/Left", "3.0": "Tapping/Right"}
)
unwanted = np.nonzero(raw_intensity.annotations.description == "15.0")
raw_intensity.annotations.delete(unwanted)


# %%
# Selecting channels appropriate for detecting neural responses
# -------------------------------------------------------------
#
# First we remove channels that are too close together (short channels) to
# detect a neural response (less than 1 cm distance between optodes).
# These short channels can be seen in the figure above.
# To achieve this we pick all the channels that are not considered to be short.

picks = mne.pick_types(raw_intensity.info, meg=False, fnirs=True)
dists = mne.preprocessing.nirs.source_detector_distances(
    raw_intensity.info, picks=picks
)
raw_intensity.pick(picks[dists > 0.01])


# %%
# Converting from raw intensity to optical density
# ------------------------------------------------
#
# The raw intensity values are then converted to optical density.

raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)


# %%
# Evaluating the quality of the data
# ----------------------------------
#
# At this stage we can quantify the quality of the coupling
# between the scalp and the optodes using the scalp coupling index. This
# method looks for the presence of a prominent synchronous signal in the
# frequency range of cardiac signals across both photodetected signals.
#
# In this example the data is clean and the coupling is good for all
# channels, so we will not mark any channels as bad based on the scalp
# coupling index.

sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od)


# %%
# In this example we will mark all channels with a SCI less than 0.5 as bad
# (this dataset is quite clean, so no channels are marked as bad).

raw_od.info["bads"] = list(compress(raw_od.ch_names, sci < 0.5))


# %%
# At this stage it is appropriate to inspect your data
# (for instructions on how to use the interactive data visualisation tool
# see :ref:`tut-visualize-raw`)
# to ensure that channels with poor scalp coupling have been removed.
# If your data contains lots of artifacts you may decide to apply
# artifact reduction techniques as described in :ref:`ex-fnirs-artifacts`.


# %%
# Converting from optical density to haemoglobin
# ----------------------------------------------
#
# Next we convert the optical density data to haemoglobin concentration using
# the modified Beer-Lambert law.

raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=0.1)

# %%
# Removing heart rate from signal
# -------------------------------
#
# The haemodynamic response has frequency content predominantly below 0.5 Hz.
# An increase in activity around 1 Hz can be seen in the data that is due to
# the person's heart beat and is unwanted. So we use a low pass filter to
# remove this. A high pass filter is also included to remove slow drifts
# in the data.

raw_haemo_unfiltered = raw_haemo.copy()
raw_haemo.filter(0.05, 0.7, h_trans_bandwidth=0.2, l_trans_bandwidth=0.02)


# %%
# Extract epochs
# --------------
#
# Now that the signal has been converted to relative haemoglobin concentration,
# and the unwanted heart rate component has been removed, we can extract epochs
# related to each of the experimental conditions.
#
# First we extract the events of interest and visualise them to ensure they are
# correct.

events, event_dict = mne.events_from_annotations(raw_haemo)

# %%
# Next we define the range of our epochs, the rejection criteria,
# baseline correction, and extract the epochs. We visualise the log of which
# epochs were dropped.

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

tapping = epochs["Tapping"].get_data()
control = epochs["Control"].get_data()

dimTappingArray = tapping.shape[0]
dimControlArray = control.shape[0]
jointArray = np.concatenate((tapping,control),axis = 0)
stopTime = 9
startTime = 11
freq = 7.81
jointArray = jointArray[:,:,int(np.floor(startTime * freq)):int(np.floor(stopTime * freq))] 