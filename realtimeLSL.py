import pickle
import numpy as np
import glob, sys, time
from pylsl import StreamInlet, resolve_byprop
from sklearn.preprocessing import StandardScaler

SAMPLE_RATE = 250
selected_channels = [0, 3, 6]
window_length_sec = 0.2
window_length = int(window_length_sec * SAMPLE_RATE)

# Load Pre-trained Model and Scaler
with open('trained_model.pkl', 'rb') as f:
    model, scaler = pickle.load(f)

# Feature Extraction Function
def compute_features(signal):
    mav = np.mean(np.abs(signal))
    ssc_count = sum(1 for i in range(1, len(signal) - 1)
                    if (signal[i] - signal[i-1]) * (signal[i+1] - signal[i]) < 0)
    zc_count = sum(1 for i in range(len(signal) - 1)
                   if (signal[i] > 0 and signal[i+1] < 0) or (signal[i] < 0 and signal[i+1] > 0))
    wl = np.sum(np.abs(np.diff(signal)))
    return np.array([mav, ssc_count, zc_count, wl])

# Resolve LSL Stream
print("Looking for an EEG LSL stream...")
streams = resolve_byprop('name', 'test', timeout=5)
if not streams:
    print("No LSL stream found. Exiting...")
    exit(1)

inlet = StreamInlet(streams[0])

def collect_fixed_samples(num_samples):
    """Collects 'num_samples' from the LSL stream and returns (n_channels, num_samples)."""
    collected = []
    while len(collected) < num_samples:
        chunk, _ = inlet.pull_chunk()
        if chunk:
            collected.extend(chunk)
    return np.array(collected).T[:, :num_samples]

# Label Mapping
label_map = {0: "fist", 1: "flat", 2: "okay", 3: "two", 4: "rest"}

# === Real-Time Classification ===
try:
    print("Real-time classification started. Press Ctrl+C to stop.")
    while True:
        # Collect real-time EMG data
        emg_data = collect_fixed_samples(window_length)
        emg_data = emg_data[selected_channels, :]

        # Extract features from each channel
        features = []
        for ch_signal in emg_data:
            features.extend(compute_features(ch_signal))
        features = np.array(features).reshape(1, -1)

        # Standardize features
        features_scaled = scaler.transform(features)

        # Predict pose
        pose_prediction = model.predict(features_scaled)[0]
        print(f"Predicted Pose: {label_map[pose_prediction]}")

except KeyboardInterrupt:
    print("\nStopping real-time classification...")
