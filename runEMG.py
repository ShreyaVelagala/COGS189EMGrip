from psychopy import visual, core, event
import numpy as np
import os
import pickle
import glob, sys, time, serial
#from brainflow.board_shim import BoardShim, BrainFlowInputParams
from serial import Serial

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, DepthwiseConv2D, SeparableConv2D, AveragePooling2D, Dropout, Flatten, Dense, Activation
from tensorflow.keras.constraints import max_norm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



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
cyton_in = False  # Set to True when using the Cyton board
subject = 1
session = 1

# Define experiment structure:
n_rounds_per_pose = 2   # Number of rounds per hand pose
n_cycles = 5            # Number of rest/hand cycles per round

rest_duration = 5       # Duration in seconds for the rest period
trial_duration = 5      # Duration in seconds for the hand pose period (recording period)
reaction_time = 0.6     # Seconds to discard at the beginning of each trial (to account for reaction/movement time)
sample_rate = 250       # OpenBCI board sampling rate (Hz)

save_dir = f'data/emg_handposes/sub-{subject:02d}/ses-{session:02d}/'
os.makedirs(save_dir, exist_ok=True)
save_file = os.path.join(save_dir, 'emg_trial_data.pkl')

# List of hand poses (and corresponding image names)
hand_poses = ['fist', 'flat', 'okay', 'two']

# =======================
# Setup PsychoPy Window
# =======================
win = visual.Window(size=(800, 600), fullscr=False, color='grey')

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


# =======================
# Main Experiment Loop
# =======================
# trial_results will be a list of dictionaries. Each dictionary represents one recorded period.
# For rest periods, the "pose" field is "Rest". For hand pose periods, it is the specific hand pose.
trial_results = []

try:
    # Loop through each hand pose
    for pose in hand_poses:
        # For each hand pose, run a number of rounds
        for rnd in range(1, n_rounds_per_pose + 1):
            # For each round, run a number of cycles
            for cycle in range(1, n_cycles + 1):
                # ===== REST PERIOD RECORDING =====
                if cyton_in:
                    board.get_board_data()  # flush the board's buffer
                rest_text.draw()
                win.flip()
                rest_start = core.getTime()
                wait_with_esc(rest_duration)
                if cyton_in:
                    rest_data = board.get_board_data()  # shape: (n_channels, n_samples)
                    discard_samples = int(reaction_time * sample_rate)
                    if rest_data.shape[1] > discard_samples:
                        rest_data = rest_data[:, discard_samples:]
                    else:
                        rest_data = np.empty((rest_data.shape[0], 0))
                else:
                    rest_data = None
                trial_data_rest = {
                    "pose": "rest",
                    "round": rnd,
                    "cycle": cycle,
                    "start_time": rest_start,
                    "duration": rest_duration,
                    "data": rest_data
                }
                trial_results.append(trial_data_rest)
                
                # ===== HAND POSE RECORDING =====
                if cyton_in:
                    board.get_board_data()  
                image_stimuli[pose].draw()
                win.flip()
                hand_start = core.getTime()
                wait_with_esc(trial_duration)
                if cyton_in:
                    hand_data = board.get_board_data()  # shape: (n_channels, n_samples)
                    discard_samples = int(reaction_time * sample_rate)
                    if hand_data.shape[1] > discard_samples:
                        hand_data = hand_data[:, discard_samples:]
                    else:
                        hand_data = np.empty((hand_data.shape[0], 0))
                else:
                    hand_data = None
                trial_data_hand = {
                    "pose": pose,
                    "round": rnd,
                    "cycle": cycle,
                    "start_time": hand_start,
                    "duration": trial_duration,
                    "data": hand_data
                }
                trial_results.append(trial_data_hand)
                

                
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

# Save the collected data only if the experiment finished normally
with open(save_file, 'wb') as f:
    pickle.dump(trial_results, f)

print("Experiment finished normally. Data saved to:", save_file)


# =======================
# Offline Classification Using EEGNet(No windowing)
# =======================
if cyton_in:
    # Build the dataset from trial_results
    # Assume each trial's data has shape (n_channels, n_samples). For our paradigm,
    X_list = []
    y_list = []
    # Map labels to integers
    label_map = {"fist": 0, "flat": 1, "okay": 2, "two": 3, "rest": 4}
    
    for trial in trial_results:
        data = trial["data"]
        if data is not None and data.size > 0:
            X_list.append(data)
            y_list.append(label_map[trial["pose"]])
    
    X = np.array(X_list)  # expected shape: (n_trials, n_channels, n_samples)
    y = np.array(y_list)
    
    # Add a singleton channel dimension so that X has shape (n_trials, n_channels, n_samples, 1) for EEGNet convolutions
    X = X[..., np.newaxis]
    
    print("Dataset shape:", X.shape)
    print("Labels shape:", y.shape)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # EEGNet implementation 
    def EEGNet(nb_classes, Chans, Samples, dropoutRate=0.5, kernLength=64, F1=8, D=2, F2=16, dropoutType='Dropout'):
        if dropoutType == 'SpatialDropout2D':
            dropoutTypeLayer = tf.keras.layers.SpatialDropout2D
        else:
            dropoutTypeLayer = tf.keras.layers.Dropout

        input1 = Input(shape=(Chans, Samples, 1))
        block1 = Conv2D(F1, (1, kernLength), padding='same', use_bias=False)(input1)
        block1 = BatchNormalization()(block1)
        block1 = DepthwiseConv2D((Chans, 1), use_bias=False, depth_multiplier=D,
                                 depthwise_constraint=max_norm(1.))(block1)
        block1 = BatchNormalization()(block1)
        block1 = Activation('elu')(block1)
        block1 = AveragePooling2D((1, 4))(block1)
        block1 = dropoutTypeLayer(dropoutRate)(block1)

        block2 = SeparableConv2D(F2, (1, 16), padding='same', use_bias=False)(block1)
        block2 = BatchNormalization()(block2)
        block2 = Activation('elu')(block2)
        block2 = AveragePooling2D((1, 8))(block2)
        block2 = dropoutTypeLayer(dropoutRate)(block2)

        flatten = Flatten()(block2)
        dense = Dense(nb_classes, kernel_constraint=max_norm(0.25))(flatten)
        softmax = Activation('softmax')(dense)

        return Model(inputs=input1, outputs=softmax)
    
    nb_classes = len(label_map)
    Chans = X.shape[1]      # number of channel
    Samples = X.shape[2]    
    
    model = EEGNet(nb_classes, Chans, Samples)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Train EEGNet for a few epochs
    print("Training EEGNet...")
    model.fit(X_train, y_train, epochs=10, batch_size=16, validation_split=0.2, verbose=1)
    
    # Evaluate on the test set
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print("Test accuracy: {:.2f}%".format(test_acc * 100))
else:
    print("No data available for classification (cyton_in is False).")