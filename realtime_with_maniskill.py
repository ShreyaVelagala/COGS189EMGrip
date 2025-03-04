import pickle
import numpy as np
import glob, sys, time
from brainflow.board_shim import BoardShim, BrainFlowInputParams
from serial import Serial
from sklearn.preprocessing import StandardScaler
from scipy.signal import iirnotch, filtfilt
import os
import time
import mujoco
import mujoco.viewer
import numpy as np
import mediapy as media
import matplotlib.pyplot as plt
np.set_printoptions(precision=3, suppress=True, linewidth=100)

# Hand Animation Joint Map Setup

model = mujoco.MjModel.from_xml_path("Adroit_hand.xml")
data = mujoco.MjData(model)
# Debug print to understand model structure
print(f"Number of actuators (controls): {model.nu}")
print("Actuator names:", [model.actuator(i).name for i in range(model.nu)])

joint_map = {}
for i in range(model.nu):
    joint_id = model.actuator(i).trnid[0]
    joint_name = model.joint(joint_id).name
    joint_map[joint_name] = i

print("\nJoint mapping:")
for name, idx in joint_map.items():
    print(f"{name}: {idx}")

# Hand Animation functions setup
def get_desired_configuration(command):
    desired_q = np.zeros(model.nu)
    try:
        if command.lower() == "rest":
            desired_q[:] = 0.0

        elif command.lower() == "fist":
            # Close all fingers
            for name in joint_map:
                if "FJ2" in name:
                    desired_q[joint_map[name]] = 1.6
                elif "FJ1" in name:
                    desired_q[joint_map[name]] = 1.0
                elif "THJ3" in name:
                    desired_q[joint_map[name]] = 1.3
                elif "THJ0" in name:
                    desired_q[joint_map[name]] = -1.57
                elif "THJ1" in name:
                    desired_q[joint_map[name]] = -0.52
                elif "THJ2" in name:
                    desired_q[joint_map[name]] = 0.1
                else:
                    desired_q[joint_map[name]] = 0.0


        elif command.lower() == "okay":
            for name in joint_map:
                # Index finger flexion
                if "FFJ1" in name:
                    desired_q[joint_map[name]] = 1.6
                elif "FFJ2" in name:
                    desired_q[joint_map[name]] = 1.0
                # Thumb opposition
                elif "THJ4" in name:
                    desired_q[joint_map[name]] = 0.2  # Abduction
                elif "THJ3" in name:
                    desired_q[joint_map[name]] = 1.0  # Flexion
                elif "THJ0" in name:
                    desired_q[joint_map[name]] = -1.0  # Tip flexion
                elif "THJ1" in name:
                    desired_q[joint_map[name]] = -0.075
                elif "THJ2" in name:
                    desired_q[joint_map[name]] = 0.135
                else:
                    desired_q[joint_map[name]] = 0.0
        elif command.lower() == "two":
            for name in joint_map:
                if "FFJ3" in name:
                    desired_q[joint_map[name]] = 0.44
                elif "MFJ3" in name:
                    desired_q[joint_map[name]] = -0.44
                elif "RFJ1" in name:
                    desired_q[joint_map[name]] = 1.6
                elif "RFJ2" in name:
                    desired_q[joint_map[name]] = 1.6
                elif "LFJ2" in name:
                    desired_q[joint_map[name]] = 1.6
                elif "LFJ1" in name:
                    desired_q[joint_map[name]] = 1.6
                elif "THJ4" in name:
                    desired_q[joint_map[name]] = 1.0
                elif "THJ3" in name:
                    desired_q[joint_map[name]] = 1.3
                elif "THJ2" in name:
                    desired_q[joint_map[name]] = 0.3
                elif "THJ1" in name:
                    desired_q[joint_map[name]] = -0.5
                elif "THJ0" in name:
                    desired_q[joint_map[name]] = -1.57
                else:
                    desired_q[joint_map[name]] = 0.0
        elif command.lower() == "flat":
            for name in joint_map:
                if "FFJ3" in name:
                    desired_q[joint_map[name]] = 0.44
                elif "MFJ3" in name:
                    desired_q[joint_map[name]] = 0.09
                elif "RFJ3" in name:
                    desired_q[joint_map[name]] = -0.4
                elif "LFJ3" in name:
                    desired_q[joint_map[name]] = -0.44
                else:
                    desired_q[joint_map[name]] = 0.0

        else:
            return None


    except KeyError as e:
        print(f"Missing joint in configuration: {e}")
        return None

    return desired_q

def show_hand(command):
    print(command)
    frames = []

    if command.lower() == "exit":
        return

    q_desired = get_desired_configuration(command)
    if q_desired is None:
        print("Invalid command")
        return

    # Reset simulation
    mujoco.mj_resetData(model, data)

    # Create renderer
    renderer = mujoco.Renderer(model, height=480, width=640)

    # Animation parameters
    duration = 2.0  # Time to reach the desired configuration
    start_time = time.time()
    frames = []  # Reset frames for new command

    # Initial joint positions
    q_start = data.qpos[:model.nu].copy()

    while time.time() - start_time < duration:
        # Interpolation factor (0 to 1)
        alpha = (time.time() - start_time) / duration
        alpha = min(alpha, 1.0)  # Clamp to 1.0

        # Interpolate between start and desired positions
        data.qpos[:model.nu] = q_start + alpha * (q_desired - q_start)

        # Update model state
        mujoco.mj_forward(model, data)

        # Render frame
        renderer.update_scene(data, camera=-1)
        frame = renderer.render()
        frames.append(frame)

    # Show video of the action
    media.show_video(frames, fps=60)
    time.sleep(5)  # Pause before next command

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
        # Calling hand animation
        show_hand(pose_prediction)

except KeyboardInterrupt:
    print("\nStopping real-time classification...")

finally:
    board.stop_stream()
    board.release_session()
