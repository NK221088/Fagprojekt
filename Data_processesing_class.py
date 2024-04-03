from itertools import compress
import matplotlib.pyplot as plt
import numpy as np
import mne
import mne_nirs
import mne_bids
import os


class fNIRS_data_load:
    def __init__(self, file_path, number_of_participants=1, annotation_names=None, stimulus_duration=5,
                 short_channel_correction=True, negative_correlation_enhancement=True, scalp_coupling_threshold=0.8,
                 reject_criteria=dict(hbo=80e-6), tmin=-5, tmax=15, baseline=(None, 0), data_types=[], number_of_data_types=2,
                 data_name="None"):
        self.number_of_participants = number_of_participants
        self.file_path = file_path
        self.annotation_names = annotation_names
        self.stimulus_duration = stimulus_duration
        self.short_channel_correction = short_channel_correction
        self.negative_correlation_enhancement = negative_correlation_enhancement
        self.scalp_coupling_threshold = scalp_coupling_threshold
        self.reject_criteria = reject_criteria
        self.tmin = tmin
        self.tmax = tmax
        self.baseline = baseline
        self.all_epochs = []
        self.all_control = []
        self.data_types = data_types
        self.number_of_data_types = len(data_types)
        self.data_name = data_name
        for name in self.data_types:
            setattr(self, f'all_{name}', [])

    def define_raw_intensity(self, sub_id):
        fnirs_data_folder = mne.datasets.fnirs_motor.data_path()
        fnirs_cw_amplitude_dir = fnirs_data_folder / "Participant-1"
        raw_intensity = mne.io.read_raw_nirx(fnirs_cw_amplitude_dir, verbose=True)
        raw_intensity.load_data()
        return raw_intensity

    def load_data(self):
        for i in range(1, self.number_of_participants + 1):
            sub_id = str(i).zfill(2)  # Pad with zeros to get "01", "02", etc.
            raw_intensity = self.define_raw_intensity(sub_id)

            raw_intensity.annotations.set_durations(5)
            raw_intensity.annotations.rename(self.annotation_names)
            unwanted = np.nonzero(raw_intensity.annotations.description == "15.0")
            raw_intensity.annotations.delete(unwanted)

            if self.short_channel_correction:
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

            raw_od.info["bads"] = list(compress(raw_od.ch_names, sci < self.scalp_coupling_threshold))

            raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=0.1)

            raw_haemo_unfiltered = raw_haemo.copy()
            raw_haemo.filter(0.05, 0.7, h_trans_bandwidth=0.2, l_trans_bandwidth=0.02)

            if self.negative_correlation_enhancement:
                raw_haemo = mne_nirs.signal_enhancement.enhance_negative_correlation(raw_haemo)

            events, event_dict = mne.events_from_annotations(raw_haemo)

            events, event_dict = mne.events_from_annotations(raw_haemo)

            epochs = mne.Epochs(
                raw_haemo,
                events,
                event_id=event_dict,
                tmin=self.tmin,
                tmax=self.tmax,
                reject=self.reject_criteria,
                reject_by_annotation=True,
                proj=True,
                baseline=self.baseline,
                preload=True,
                detrend=None,
                verbose=True,
            )

            self.all_epochs.append(epochs)
            self.all_control.append(epochs["Control"].get_data())
            for name in self.data_types:
                getattr(self, f'all_{name}').append(epochs[name].get_data())

        # Concatenate the control data
        self.all_control = np.concatenate(self.all_control, axis=0)

        # Concatenate the data for each data type
        for name in self.data_types:
            setattr(self, f'all_{name}', np.concatenate(getattr(self, f'all_{name}'), axis=0))

        # Create the dictionary all_data with Control and data for each data type
        all_data = {"Control": self.all_control}
        for name in self.data_types:
            all_data.update({name: getattr(self, f'all_{name}')})

        # Update all_data with control_dict
        all_freq = self.all_epochs[0].info['sfreq']
        self.data_types.append("Control")
        return self.all_epochs, self.data_name, all_data, all_freq, self.data_types

        
class AudioSpeechNoise_data_load(fNIRS_data_load):
    def __init__(self, short_channel_correction : bool, negative_correlation_enhancement : bool):
        self.number_of_participants = 17
        self.all_speech = []
        self.all_noise = []
        self.annotation_names = {"1.0": "Control",
                            "2.0": "Activity/Noise",
                            "3.0": "Activity/Speech"}
        self.file_path = mne_nirs.datasets.block_speech_noise.data_path()
        self.short_channel_correction = short_channel_correction
        self.negative_correlation_enhancement = negative_correlation_enhancement
        self.stimulus_duration = 5
        self.scalp_coupling_threshold = 0.5
        self.reject_criteria = dict(hbo=80e-6)
        self.tmin = -5
        self.tmax = 15
        self.baseline = (None, 0)
        self.data_types = ["Speech", "Noise"]
        self.number_of_data_types = 2
        self.data_name = "AudioSpeechNoise"
        super().__init__(
                        number_of_participants = self.number_of_participants,
                        file_path = self.file_path,
                        annotation_names = self.annotation_names,
                        stimulus_duration = self.stimulus_duration,
                        short_channel_correction = self.short_channel_correction,
                        negative_correlation_enhancement = self.negative_correlation_enhancement,
                        scalp_coupling_threshold = self.scalp_coupling_threshold,
                        reject_criteria = self.reject_criteria,
                        baseline = self.baseline,
                        tmin = self.tmin,
                        tmax = self.tmax,
                        data_types = self.data_types,
                        number_of_data_types = self.number_of_data_types,
                        data_name = self.data_name)

    def define_raw_intensity(self, sub_id):
        fnirs_snirf_file_path = os.path.join(self.file_path, f"sub-{sub_id}", "ses-01", "nirs", f"sub-{sub_id}_ses-01_task-AudioSpeechNoise_nirs.snirf")
        raw_intensity = mne.io.read_raw_snirf(fnirs_snirf_file_path, verbose=True)
        raw_intensity.load_data()
        return raw_intensity

class fNIRS_motor_data_load(fNIRS_data_load):
    def __init__(self, short_channel_correction: bool, negative_correlation_enhancement: bool):
        self.number_of_participants = 1
        self.all_tapping = []
        self.all_control = []
        self.annotation_names = {"1.0": "Control",
                                "2.0": "Tapping/Left",
                                "3.0": "Tapping/Right"}
        self.file_path = mne.datasets.fnirs_motor.data_path()
        self.short_channel_correction = short_channel_correction
        self.negative_correlation_enhancement = negative_correlation_enhancement
        self.stimulus_duration = 5
        self.scalp_coupling_threshold = 0.5  # Change this value if needed
        self.reject_criteria = dict(hbo=80e-6)
        self.tmin = -5
        self.tmax = 15
        self.baseline = (None, 0)
        self.data_types = ["Tapping"]
        self.number_of_data_types = 2
        self.data_name = "fnirs_motor_plus_anti"
        super().__init__(
            number_of_participants=self.number_of_participants,
            file_path=self.file_path,
            annotation_names=self.annotation_names,
            stimulus_duration=self.stimulus_duration,
            short_channel_correction=self.short_channel_correction,
            negative_correlation_enhancement=self.negative_correlation_enhancement,
            scalp_coupling_threshold=self.scalp_coupling_threshold,
            reject_criteria=self.reject_criteria,
            baseline=self.baseline,
            tmin=self.tmin,
            tmax=self.tmax,
            data_types=self.data_types,
            number_of_data_types=self.number_of_data_types,
            data_name=self.data_name
        )

    def define_raw_intensity(self, sub_id):
        fnirs_data_folder = mne.datasets.fnirs_motor.data_path()
        fnirs_cw_amplitude_dir = fnirs_data_folder / "Participant-1"
        raw_intensity = mne.io.read_raw_nirx(fnirs_cw_amplitude_dir, verbose=True)
        raw_intensity.load_data()
        return raw_intensity

class fNIRS_full_motor_data_load(fNIRS_data_load):
    def __init__(self, short_channel_correction: bool, negative_correlation_enhancement: bool):
        self.number_of_participants = 5
        self.all_tapping = []
        self.all_control = []
        self.annotation_names = {"1.0": "Control",
                                 "2.0": "Tapping/Left",
                                 "3.0": "Tapping/Right"}
        self.file_path = mne.datasets.fnirs_motor.data_path()
        self.short_channel_correction = short_channel_correction
        self.negative_correlation_enhancement = negative_correlation_enhancement
        self.stimulus_duration = 5
        self.scalp_coupling_threshold = 0.5  # Change this value if needed
        self.reject_criteria = dict(hbo=80e-6)
        self.tmin = -5
        self.tmax = 15
        self.baseline = (None, 0)
        self.data_types = ["Tapping"]
        self.number_of_data_types = 2
        self.data_name = "fnirs_full_motor"
        super().__init__(
            number_of_participants=self.number_of_participants,
            file_path=self.file_path,
            annotation_names=self.annotation_names,
            stimulus_duration=self.stimulus_duration,
            short_channel_correction=self.short_channel_correction,
            negative_correlation_enhancement=self.negative_correlation_enhancement,
            scalp_coupling_threshold=self.scalp_coupling_threshold,
            reject_criteria=self.reject_criteria,
            baseline=self.baseline,
            tmin=self.tmin,
            tmax=self.tmax,
            data_types=self.data_types,
            number_of_data_types=self.number_of_data_types,
            data_name=self.data_name
        )

    def define_raw_intensity(self, sub_id):
        raw_intensity = mne.io.read_raw_snirf(f"Dataset/rob-luke/rob-luke-BIDS-NIRS-Tapping-e262df8/sub-{sub_id}/nirs/sub-{sub_id}_task-tapping_nirs.snirf", verbose=True)
        raw_intensity.load_data()
        return raw_intensity