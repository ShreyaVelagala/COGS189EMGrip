from psychopy import visual, core, event
import numpy as np
import os
import pickle
import glob, sys, time, serial
from brainflow.board_shim import BoardShim, BrainFlowInputParams
from serial import Serial
from scipy.signal import iirnotch, filtfilt  # for notch filtering



class EscapeException(Exception):
    pass

def wait_with_esc(duration):
    """Wait for 'duration' seconds while checking for the escape key.
    If escape is pressed, raise an EscapeException to abort the experiment."""
    start = core.getTime()
    while core.getTime() - start < duration:
        if 'escape' in event.getKeys(keyList=["escape"]):
            raise EscapeException
        core.wait(0.01)  # Check every 10 ms




# =======================
# Experiment Parameters
# =======================
cyton_in = True
subject = 1
session = 1
sampling_rate = 250
reaction_time = 0.6  # 600 ms
discard_samples = int(reaction_time * sampling_rate)  # e.g. 150 samples

n_rounds_per_pose = 2
n_cycles = 5
rest_duration_seconds = 5
trial_duration_seconds = 5
rest_samples = int(rest_duration_seconds * sampling_rate)   
trial_samples = int(trial_duration_seconds * sampling_rate) 


save_dir = f'data/emg_handposes/sub-{subject:02d}/ses-{session:02d}/'
os.makedirs(save_dir, exist_ok=True)
save_file = os.path.join(save_dir, 'emg_trial_data.pkl')

# List of hand poses (and corresponding image names)
hand_poses = ['fist', 'flat', 'okay', 'two']

# =======================
# Setup PsychoPy Window
# =======================
win = visual.Window(size=(800, 600), fullscr=False,  units="norm", color='grey')

# Create a stimulus for the rest period
rest_text = visual.TextStim(win, text='Rest', height=0.1, color='white', pos=(0, 0))

# Preload image stimuli for each hand pose from the "positions" folder.
image_stimuli = {}
for pose in hand_poses:
    image_path = os.path.join("positions", f"{pose}.jpg")
    image_stimuli[pose] = visual.ImageStim(win, image=image_path, size=(0.4, 0.8))

# =======================
# Setup OpenBCI Cyton Board (if in use)
# =======================
if cyton_in:
    def find_openbci_port():
        """Finds the serial port to which the OpenBCI Cyton is connected."""
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
                if s.inWaiting():
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
    board.start_stream(45000)

def collect_fixed_samples(num_samples):
    """
    Continuously poll board.get_board_data() until 'num_samples' is reached.
    Returns the concatenated data array of shape (n_channels, num_samples).
    """
    collected = np.zeros((8, 0))  # 8 channels, 0 samples to start
    while True:
        new_data = board.get_board_data()  # shape: (n_channels, n_new)
        eeg_data = new_data[board.get_eeg_channels(CYTON_BOARD_ID)]
        if new_data.size > 0:
            if collected.size == 0:
                collected = eeg_data
            else:
                collected = np.concatenate((collected, eeg_data), axis=1)
        #print("collected.shape:", collected.shape)
        if collected.shape[1] >= num_samples:
            break
        core.wait(0.01)  # small pause to avoid busy loop
    return collected[:, :num_samples]

# =======================
# Main Experiment Loop
# =======================
# trial_results will be a list of dictionaries. Each dictionary represents one recorded period.
# For rest periods, the "pose" field is "Rest". For hand pose periods, it is the specific hand pose.
trial_results = []

try:
    for pose in hand_poses:
        for rnd in range(1, n_rounds_per_pose + 1):
            for cycle in range(1, n_cycles + 1):
                # --- Rest Period (Display) ---
                rest_text.draw()
                win.flip()
                #wait_with_esc(rest_duration_seconds)
                # --- Rest Data ---
                if cyton_in:
                    board.get_board_data()  # flush buffer
                    rest_data = collect_fixed_samples(rest_samples)
                else:
                    rest_data = None

                trial_data_rest = {
                    "pose": "Rest",
                    "round": rnd,
                    "cycle": cycle,
                    "samples": rest_samples,
                    "data": rest_data
                }
                trial_results.append(trial_data_rest)

                # --- Pose Period (Display) ---
                image_stimuli[pose].draw()
                win.flip()
                #wait_with_esc(trial_duration_seconds)

                # --- Pose Data ---
                if cyton_in:
                    board.get_board_data()  # flush buffer
                    pose_data = collect_fixed_samples(trial_samples)
                else:
                    pose_data = None

                trial_data_pose = {
                    "pose": pose,
                    "round": rnd,
                    "cycle": cycle,
                    "samples": trial_samples,
                    "data": pose_data
                }
                trial_results.append(trial_data_pose)
             

                
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


for trial in trial_results:
    data = trial["data"]
    if data is not None and data.shape[1] > discard_samples:
        data = data[:, discard_samples:1200]
    else:
        # either not enough data or None
        data = np.empty((data.shape[0], 0)) if data is not None else None
    trial["data"] = data

# Save the collected data only if the experiment finished normally
with open(save_file, 'wb') as f:
    pickle.dump(trial_results, f)

print("Experiment finished normally. Data saved to:", save_file)
