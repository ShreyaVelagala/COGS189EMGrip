from psychopy import visual, core, event
import numpy as np
import os
import pickle
import glob, sys, time, serial
from brainflow.board_shim import BoardShim, BrainFlowInputParams
from serial import Serial
from scipy.signal import iirnotch, butter, lfilter, lfilter_zi



class EscapeException(Exception):
    pass



cyton_in = True
subject = 1
session = 1
SAMPLE_RATE = 250
CUTOFF_FREQ = 1.0  # Hz high-pass cutoff

reaction_time = 0.6  # seconds to discard initially (calibration)
discard_samples = int(reaction_time * SAMPLE_RATE)  # e.g. 150 samples

n_rounds_per_pose = 2
n_cycles = 5
rest_duration_seconds = 5
trial_duration_seconds = 5
rest_samples = int(rest_duration_seconds * SAMPLE_RATE)
trial_samples = int(trial_duration_seconds * SAMPLE_RATE)

save_dir = f'data/emg_handposes/sub-{subject:02d}/ses-{session:02d}/'
os.makedirs(save_dir, exist_ok=True)
save_file = os.path.join(save_dir, 'emg_trial_data.pkl')

# List of hand poses (and corresponding image names)
hand_poses = ['fist', 'flat', 'okay', 'two']

# ------------------------------
# Setup PsychoPy Window and Stimuli
# ------------------------------
win = visual.Window(size=(800, 600), fullscr=False, units="norm", color='grey')
rest_text = visual.TextStim(win, text='Rest', height=0.1, color='white', pos=(0, 0))
image_stimuli = {}
for pose in hand_poses:
    image_path = os.path.join("positions", f"{pose}.jpg")
    image_stimuli[pose] = visual.ImageStim(win, image=image_path, size=(0.4, 0.8))

# ------------------------------
# Setup OpenBCI Cyton Board (if in use)
# ------------------------------
if cyton_in:
    def find_openbci_port():
        if sys.platform.startswith('win'):
            ports = ['COM%s' % (i + 1) for i in range(256)]
        elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
            ports = glob.glob('/dev/ttyUSB*')
        elif sys.platform.startswith('darwin'):
            ports = glob.glob('/dev/cu.usbserial*')
        else:
            raise EnvironmentError('Error finding ports on your operating system')
        for port in ports:
            try:
                s = Serial(port=port, baudrate=115200, timeout=1)
                s.write(b'v')
                time.sleep(2)
                if s.in_waiting:
                    line = ''
                    while '$$$' not in line:
                        line += s.read().decode('utf-8', errors='replace')
                    if 'OpenBCI' in line:
                        s.close()
                        return port
                s.close()
            except (OSError, serial.SerialException):
                pass
        raise OSError('Cannot find OpenBCI port.')
    
    params = BrainFlowInputParams()
    params.serial_port = find_openbci_port()
    CYTON_BOARD_ID = 0
    board = BoardShim(CYTON_BOARD_ID, params)
    board.prepare_session()
    board.start_stream()

def collect_fixed_samples(num_samples):
    """
    Continuously poll board.get_board_data() until 'num_samples' is reached.
    Returns concatenated data of shape (n_channels, num_samples).
    """
    collected = np.zeros((8, 0))  # assume 8 channels
    while collected.shape[1] < num_samples:
        new_data = board.get_board_data()  # shape: (n_channels, n_new)
        eeg_data = new_data[board.get_eeg_channels(CYTON_BOARD_ID)]
        if eeg_data.size > 0:
            collected = np.concatenate((collected, eeg_data), axis=1)
        core.wait(0.01)
    return collected[:, :num_samples]

# ------------------------------
# Initialize Filter Coefficients and States for Continuous Filtering
# ------------------------------

n_channels = 8

# Setup comb notch filters (for 60 Hz and 120 Hz)
n_harmonics = 2
comb_filters = []
comb_states = []  # one state array per harmonic

for i in range(1, n_harmonics+1):
    freq = i * 60.0
    if freq < SAMPLE_RATE / 2:
        b, a = iirnotch(freq, 30, SAMPLE_RATE)
        comb_filters.append((b, a))
        # Compute initial conditions for n_channels
        zi = lfilter_zi(b, a)
        comb_states.append(np.tile(zi, (n_channels, 1)))  # shape: (n_channels, filter_order)

# Setup high-pass filter (Butterworth, order=4)
b_hp, a_hp = butter(4, CUTOFF_FREQ / (0.5 * SAMPLE_RATE), btype='high', analog=False)
hp_state = np.tile(lfilter_zi(b_hp, a_hp), (n_channels, 1))  # shape: (n_channels, filter_order)

# ------------------------------
# Calibration Data Collection Loop with Continuous (Stateful) Filtering
# ------------------------------
trial_results = []

try:
    for pose in hand_poses:
        for rnd in range(1, n_rounds_per_pose + 1):
            for cycle in range(1, n_cycles + 1):
                # --- Rest Period: Display and Collect Data ---
                rest_text.draw()
                win.flip()
                # Wait for rest duration visually (optional)
                
                if cyton_in:
                    board.get_board_data()  # flush buffer
                    raw_rest = collect_fixed_samples(rest_samples)  # shape: (8, rest_samples)
                else:
                    raw_rest = None

                # Apply continuous (stateful) filtering on raw_rest:
                if raw_rest is not None and raw_rest.shape[1] > discard_samples:
                    # Process only from discard_samples onward (to remove initial transients)
                    data_to_filter = raw_rest[:, discard_samples:]
                    
                    # Apply comb notch filters sequentially, updating their states
                    filtered_rest = data_to_filter.copy()
                    for idx, (b, a) in enumerate(comb_filters):
                        filtered_rest, new_state = lfilter(b, a, filtered_rest, axis=1, zi=comb_states[idx].T)
                        comb_states[idx] = new_state.T  # update state for all channels
                        
                    # Apply high-pass filter (update state)
                    filtered_rest, hp_state = lfilter(b_hp, a_hp, filtered_rest, axis=1, zi=hp_state.T)
                    hp_state = hp_state.T
                else:
                    filtered_rest = None
                
                trial_results.append({
                    "pose": "Rest",
                    "round": rnd,
                    "cycle": cycle,
                    "samples": rest_samples - discard_samples,
                    "data": filtered_rest
                })
                
                # --- Pose Period: Display and Collect Data ---
                image_stimuli[pose].draw()
                win.flip()
                
                if cyton_in:
                    board.get_board_data()  # flush buffer
                    raw_pose = collect_fixed_samples(trial_samples)
                else:
                    raw_pose = None
                
                if raw_pose is not None and raw_pose.shape[1] > discard_samples:
                    data_to_filter = raw_pose[:, discard_samples:]
                    
                    filtered_pose = data_to_filter.copy()
                    for idx, (b, a) in enumerate(comb_filters):
                        filtered_pose, new_state = lfilter(b, a, filtered_pose, axis=1, zi=comb_states[idx].T)
                        comb_states[idx] = new_state.T
                    filtered_pose, hp_state = lfilter(b_hp, a_hp, filtered_pose, axis=1, zi=hp_state.T)
                    hp_state = hp_state.T
                else:
                    filtered_pose = None
                
                trial_results.append({
                    "pose": pose,
                    "round": rnd,
                    "cycle": cycle,
                    "samples": trial_samples - discard_samples,
                    "data": filtered_pose
                })
                
except EscapeException:
    win.close()
    print("Escape pressed. Aborting experiment without saving data.")
    if cyton_in:
        board.stop_stream()
        board.release_session()
    core.quit()

win.close()
if cyton_in:
    board.stop_stream()
    board.release_session()

# Save the calibration data
with open(save_file, 'wb') as f:
    pickle.dump(trial_results, f)

print("Experiment finished normally. Data saved to:", save_file)