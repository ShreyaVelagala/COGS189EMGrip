import pickle
import numpy as np
import glob, sys, time
from brainflow.board_shim import BoardShim, BrainFlowInputParams
from serial import Serial
from sklearn.preprocessing import StandardScaler
from scipy.signal import iirnotch, filtfilt

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

# Label Mapping
label_map = {0: "fist", 1: "flat", 2: "okay", 3: "two", 4: "rest"}

# === Real-Time Classification ===
try:
    print("Real-time classification started. Press Ctrl+C to stop.")
    while True:
        # Collect real-time EMG data (250 samples for 1 second at 250Hz)
        emg_data = collect_fixed_samples(250)

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
