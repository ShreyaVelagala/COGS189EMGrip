import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch


class EMGDataManager:
    def __init__(self, file_path, sample_rate=250, cutoff_freq=20.0, selected_channels=None):
        self.file_path = file_path
        self.sample_rate = sample_rate
        self.cutoff_freq = cutoff_freq
        # Default channels if none are specified
        self.selected_channels = selected_channels if selected_channels is not None else [1, 4, 7]

    def load_and_group(self):
        """Load the pickle file and group trials by label."""
        import pickle
        with open(self.file_path, "rb") as f:
            trial_results = pickle.load(f)
        
        data_by_label = {}
        for trial in trial_results:
            label = trial["pose"]
            if trial["data"] is not None and trial["data"].size > 0:
                data_by_label.setdefault(label, []).append(trial["data"])
        return data_by_label

    @staticmethod
    def comb_notch_filter(data, fs, fundamental=60, n_harmonics=2, Q=30):
        """Apply sequential notch filters to remove line noise and its harmonics."""
        from scipy.signal import iirnotch, filtfilt
        import numpy as np
        filtered_data = np.copy(data)
        for i in range(1, n_harmonics + 1):
            freq = i * fundamental
            if freq < fs / 2:
                b, a = iirnotch(freq, Q, fs)
                filtered_data = filtfilt(b, a, filtered_data, axis=1)
        return filtered_data

    @staticmethod
    def highpass_filter(data, cutoff, fs, order=4):
        """Apply a Butterworth high-pass filter."""
        from scipy.signal import butter, filtfilt
        import numpy as np
        nyquist = 0.5 * fs
        normalized_cutoff = cutoff / nyquist
        b, a = butter(order, normalized_cutoff, btype='high', analog=False)
        return filtfilt(b, a, data, axis=1)
    
    def preprocess(self, data_by_label):
        """
        Apply filtering and channel selection.
        Each trial is filtered for line noise and low-frequency drift,
        then only the selected channels are kept.
        """
        for label, trials in data_by_label.items():
            for i in range(len(trials)):
                if trials[i].size > 0:
                    filtered = self.comb_notch_filter(trials[i], fs=self.sample_rate, fundamental=60, n_harmonics=2, Q=30)
                    trials[i] = self.highpass_filter(filtered, cutoff=self.cutoff_freq, fs=self.sample_rate, order=4)
                    # Select channels
                    trials[i] = trials[i][self.selected_channels, :]
        return data_by_label

    def window_data(self, data_by_label, window_length_sec=0.42, overlap=0):
        """
        Split each trial into fixed-length windows.
        Returns:
            X_windowed: NumPy array of windowed data.
            y_windowed: List of corresponding labels.
        """
        import numpy as np
        window_length = int(window_length_sec * self.sample_rate)
        step = max(1, int(window_length * (1 - overlap)))
        X_windowed, y_windowed = [], []
        for label, trials in data_by_label.items():
            for trial in trials:
                if trial.size > 0:
                    n_samples = trial.shape[1]
                    for start in range(0, n_samples - window_length + 1, step):
                        X_windowed.append(trial[:, start:start + window_length])
                        print(trial[:, start:start + window_length])
                        y_windowed.append(label)
        return np.array(X_windowed), y_windowed

    def balance_data(self, X, y, target_label, random_state=None):
        """
        Balance the dataset by downsampling the target label to match the smallest
        number of samples among the other classes.
        
        Parameters:
            X : np.array
                The array of windowed data or features.
            y : list
                The list of corresponding labels.
            target_label : str or int
                The label to downsample (e.g., "Rest" or its integer representation).
            random_state : int or None
                Optional random seed for reproducibility.
                
        Returns:
            X_balanced, y_balanced : np.array, list
                The balanced data and labels.
        """
        import random
        from collections import defaultdict
        
        if random_state is not None:
            random.seed(random_state)
        
        # Group samples by label
        class_data = defaultdict(list)
        for xi, label in zip(X, y):
            class_data[label].append(xi)
        
        # Determine the minimum sample size among classes excluding the target label
        min_other_class_size = min(len(samples)
                                   for lbl, samples in class_data.items() if lbl != target_label)
        
        X_balanced, y_balanced = [], []
        for lbl, samples in class_data.items():
            if lbl == target_label:
                # Downsample the target class to match the minimum of the other classes
                if len(samples) > min_other_class_size:
                    chosen = random.sample(samples, min_other_class_size)
                else:
                    chosen = samples
                X_balanced.extend(chosen)
                y_balanced.extend([lbl] * len(chosen))
            else:
                # Keep all samples from the other classes
                X_balanced.extend(samples)
                y_balanced.extend([lbl] * len(samples))
        
        return np.array(X_balanced), y_balanced


    @staticmethod
    def compute_features(signal):
        """
        Compute features for a 1D signal.
        Features:
            - MAV: Mean Absolute Value
            - SSC: Slope Sign Changes
            - ZC: Zero Crossings
            - WL: Waveform Length
        """
        import numpy as np
        mav = np.mean(np.abs(signal))
        ssc_count = sum(
            1 for i in range(1, len(signal) - 1)
            if (signal[i] - signal[i-1]) * (signal[i+1] - signal[i]) < 0
        )
        zc_count = sum(
            1 for i in range(len(signal) - 1)
            if (signal[i] > 0 and signal[i+1] < 0) or (signal[i] < 0 and signal[i+1] > 0)
        )
        wl = np.sum(np.abs(np.diff(signal)))
        return [mav, ssc_count, zc_count, wl]
    
    def extract_features(self, X):
        """
        Extract features for each window.
        For every window (shape: [n_channels, window_length]), compute features
        for each channel and concatenate them.
        Returns:
            all_features: NumPy array where each row is the feature vector for a window.
        """
        import numpy as np
        all_features = []
        for window in X:
            window_features = []
            for ch_signal in window:
                window_features.extend(self.compute_features(ch_signal))
            all_features.append(window_features)
        return np.array(all_features)
