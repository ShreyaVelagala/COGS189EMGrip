import pickle
import numpy as np
import glob, sys, time
from brainflow.board_shim import BoardShim, BrainFlowInputParams
from serial import Serial
from sklearn.preprocessing import StandardScaler
from scipy.signal import lfilter, lfilter_zi, iirnotch, butter


SAMPLE_RATE = 250
CUTOFF_FREQ = 1.  # Hz high-pass cutoff
selected_channels = [1, 4, 7]
window_length_sec = 0.42
window_length = int(window_length_sec* SAMPLE_RATE)

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

def comb_notch_filter(data, fs=SAMPLE_RATE, fundamental=60, n_harmonics=2, Q=30):
    """Sequentially applies notch filters to remove line noise at the fundamental and its harmonics."""
    filtered_data = np.copy(data)
    for i in range(1, n_harmonics + 1):
        freq = i * fundamental
        if freq < fs / 2:
            b, a = iirnotch(freq, Q, fs)
            filtered_data = filtfilt(b, a, filtered_data, axis=1)
    return filtered_data

def highpass_filter(data, cutoff, fs=SAMPLE_RATE, order=4):
    """Applies a Butterworth high-pass filter along axis=1."""
    nyquist = 0.5 * fs
    normalized_cutoff = cutoff / nyquist
    b, a = butter(order, normalized_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data, axis=1)

# =======================
# Setup OpenBCI Cyton Board (if in use)
# =======================
def find_openbci_port():
    """Finds the serial port for the OpenBCI Cyton board."""
    if sys.platform.startswith('win'):
        ports = ['COM%s' % (i + 1) for i in range(256)]
    elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
        ports = glob.glob('/dev/ttyUSB*')
    elif sys.platform.startswith('darwin'):
        ports = glob.glob('/dev/cu.usbserial*')
    else:
        raise EnvironmentError('Unsupported platform')

    for port in ports:
        try:
            s = Serial(port=port, baudrate=115200, timeout=1)
            s.write(b'v')
            time.sleep(2)
            if s.in_waiting:
                line = s.read(s.in_waiting).decode('utf-8', errors='replace')
                if 'OpenBCI' in line:
                    s.close()
                    return port
            s.close()
        except Exception:
            pass
    raise OSError('OpenBCI port not found')

# Configure BrainFlow
params = BrainFlowInputParams()
params.serial_port = find_openbci_port()
CYTON_BOARD_ID = 0
board = BoardShim(CYTON_BOARD_ID, params)
board.prepare_session()
board.start_stream()

# Data Collection Function
def collect_fixed_samples(num_samples):
    collected = np.zeros((8, 0))  # 8 channels, start with 0 samples
    while collected.shape[1] < num_samples:
        new_data = board.get_board_data()
        eeg_data = new_data[board.get_eeg_channels(CYTON_BOARD_ID)]
        if eeg_data.size > 0:
            collected = np.concatenate((collected, eeg_data), axis=1)
        time.sleep(0.01)  # Avoid busy loop
    return collected[:, :num_samples]

n_harmonics = 2
comb_filters = []   # list of (b, a) tuples for each harmonic
comb_states = []    # list of state arrays for each filter, one per selected channel

for i in range(1, n_harmonics + 1):
    freq = i * 60.0
    if freq < SAMPLE_RATE / 2:
        b, a = iirnotch(freq, 30, SAMPLE_RATE)
        comb_filters.append((b, a))
        # Initialize state for each selected channel
        # lfilter_zi returns a 1D array of initial conditions; replicate for each channel.
        zi = lfilter_zi(b, a)  # shape: (filter_order,)
        comb_states.append(np.tile(zi, (len(selected_channels), 1)))  # shape: (n_selected, filter_order)

# Set up high-pass filter (Butterworth, order=4)
b_hp, a_hp = butter(4, CUTOFF_FREQ / (0.5 * SAMPLE_RATE), btype='high', analog=False)
hp_state = np.tile(lfilter_zi(b_hp, a_hp), (len(selected_channels), 1))  # shape: (n_selected, filter_order)

# Label Mapping
label_map = {0: "fist", 1: "flat", 2: "okay", 3: "two", 4: "rest"}

# === Real-Time Classification ===
try:
    print("Real-time classification started. Press Ctrl+C to stop.")
    while True:
        # Collect real-time EMG data
        emg_data = collect_fixed_samples(window_length)
        emg_data = emg_data[selected_channels, :]
  
        
        filtered_data = emg_data.copy()
        
        # Apply the comb notch filters causally (stateful filtering)
        for i, (b, a) in enumerate(comb_filters):
            # lfilter with state along axis=1; note: we pass the current state for each channel
            filtered_data, new_state = lfilter(b, a, filtered_data, axis=1, zi=comb_states[i].T)
            # Update the state (transpose back to original shape)
            comb_states[i] = new_state.T
        
        # Apply the high-pass filter causally
        filtered_data, hp_state = lfilter(b_hp, a_hp, filtered_data, axis=1, zi=hp_state.T)
        hp_state = hp_state.T
        

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

finally:
    board.stop_stream()
    board.release_session()
