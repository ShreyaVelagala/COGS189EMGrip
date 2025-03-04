# hand_control_system.py
import threading
import time
import numpy as np
import joblib
import pickle
import os
import glob
import sys
# import serial
# from psychopy import visual, core, event
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
# from brainflow.board_shim import BoardShim, BrainFlowInputParams
from scipy.signal import iirnotch, filtfilt
import mujoco
import mediapy as media

from final import X_test_scaled, y_test


SAMPLE_RATE = 250
BUFFER_DURATION = 2.0  
PREDICTION_INTERVAL = 0.1  
ACTION_HOLD_DURATION = 10  
CYTON_BOARD_ID = 0

class HandControlState:
    def __init__(self):
        self.current_pose = "rest"
        self.prediction_buffer = []
        self.data_buffer = np.empty((8, 0))  # 8 EMG channels
        self.lock = threading.Lock()
        self.last_action_time = 0
        self.running = True


# def find_openbci_port():
#     if sys.platform.startswith('win'):
#         ports = ['COM%s' % (i + 1) for i in range(256)]
#     elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
#         ports = glob.glob('/dev/ttyUSB*')
#     elif sys.platform.startswith('darwin'):
#         ports = glob.glob('/dev/cu.usbserial*')
#     else:
#         raise EnvironmentError('Unsupported OS')
    
#     for port in ports:
#         try:
#             s = serial.Serial(port=port, baudrate=115200, timeout=1)
#             s.write(b'v')
#             time.sleep(2)
#             if s.inWaiting():
#                 line = ''
#                 while '$$$' not in line:
#                     line += s.read().decode('utf-8', errors='replace')
#                 if 'OpenBCI' in line:
#                     s.close()
#                     return port
#             s.close()
#         except (OSError, serial.SerialException):
#             pass
#     raise OSError('OpenBCI port not found')

# def init_cyton():
#     params = BrainFlowInputParams()
#     params.serial_port = find_openbci_port()
#     board = BoardShim(CYTON_BOARD_ID, params)
#     board.prepare_session()
#     board.start_stream(45000)
#     return board

# def collect_fixed_samples(board, num_samples):
#     collected = np.zeros((8, 0))
#     while True:
#         new_data = board.get_board_data()
#         if new_data.size > 0:
#             collected = np.concatenate((collected, new_data[1:9]), axis=1)
#         if collected.shape[1] >= num_samples:
#             return collected[:, :num_samples]
#         core.wait(0.01)


class HandAnimator:
    def __init__(self, model_path):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, 480, 640)
        self.joint_map = self._create_joint_map()
        
    def _create_joint_map(self):
        joint_map = {}
        for i in range(self.model.nu):
            joint_id = self.model.actuator(i).trnid[0]
            joint_name = self.model.joint(joint_id).name
            joint_map[joint_name] = i
        return joint_map
    
    def animate(self, command, duration=1.0):
        q_target = self._get_target_config(command)
        if q_target is None:
            return
            
        q_start = self.data.qpos.copy()
        start_time = time.time()
        
        while (time.time() - start_time) < duration:
            alpha = (time.time() - start_time) / duration
            self.data.qpos[:] = q_start + alpha * (q_target - q_start)
            mujoco.mj_forward(self.model, self.data)
            time.sleep(0.01)
            
    def _get_target_config(self, command):
        config = np.zeros(self.model.nu)
        try:
            if command == "rest":
                return config
            elif command.lower() == "fist":
                # Close all fingers
                for name in self.joint_map:
                    if "FJ2" in name:
                        config[self.joint_map[name]] = 1.6
                    elif "FJ1" in name:
                        config[self.joint_map[name]] = 1.0
                    elif "THJ3" in name:
                        config[self.joint_map[name]] = 1.3
                    elif "THJ0" in name: 
                        config[self.joint_map[name]] = -1.57
                    elif "THJ1" in name: 
                        config[self.joint_map[name]] = -0.52
                    elif "THJ2" in name:
                        config[self.joint_map[name]] = 0.1
                    else:
                        config[self.joint_map[name]] = 0.0
                return config
                        
            elif command.lower() == "okay":
                for name in self.joint_map:
                    # Index finger flexion
                    if "FFJ1" in name:
                        config[self.joint_map[name]] = 1.6
                    elif "FFJ2" in name:
                        config[self.joint_map[name]] = 1.0
                    # Thumb opposition
                    elif "THJ4" in name:
                        config[self.joint_map[name]] = 0.2  
                    elif "THJ3" in name:
                        config[self.joint_map[name]] = 1.0  
                    elif "THJ0" in name:
                        config[self.joint_map[name]] = -1.0  
                    elif "THJ1" in name:
                        config[self.joint_map[name]] = -0.075
                    elif "THJ2" in name:
                        config[self.joint_map[name]] = 0.135
                    else:
                        config[self.joint_map[name]] = 0.0
                return config
            elif command.lower() == "two":
                for name in self.joint_map:
                    if "FFJ3" in name:
                        config[self.joint_map[name]] = 0.44
                    elif "MFJ3" in name:
                        config[self.joint_map[name]] = -0.44
                    elif "RFJ1" in name:
                        config[self.joint_map[name]] = 1.6
                    elif "RFJ2" in name:
                        config[self.joint_map[name]] = 1.6
                    elif "LFJ2" in name:
                        config[self.joint_map[name]] = 1.6
                    elif "LFJ1" in name:
                        config[self.joint_map[name]] = 1.6
                    elif "THJ4" in name:
                        config[self.joint_map[name]] = 1.0
                    elif "THJ3" in name:
                        config[self.joint_map[name]] = 1.3
                    elif "THJ2" in name:
                        config[self.joint_map[name]] = 0.3
                    elif "THJ1" in name:
                        config[self.joint_map[name]] = -0.5
                    elif "THJ0" in name:
                        config[self.joint_map[name]] = -1.57
                    else:
                        config[self.joint_map[name]] = 0.0
                return config
            
            elif command.lower() == "flat":
                for name in self.joint_map:
                    if "FFJ3" in name:
                        config[self.joint_map[name]] = 0.44
                    elif "MFJ3" in name:
                        config[self.joint_map[name]] = 0.09
                    elif "RFJ3" in name:
                        config[self.joint_map[name]] = -0.4
                    elif "LFJ3" in name:
                        config[self.joint_map[name]] = -0.44
                    else:
                        config[self.joint_map[name]] = 0.0
                return config
                
            else:
                return None
        except KeyError as e:
            print(f"Configuration error: {e}")
            return None

class HandPoseClassifier:
    def __init__(self, model_path):
        assets = joblib.load(model_path)
        self.scaler = assets['pipeline'].steps[0][1]
        self.model = assets['pipeline'].steps[1][1]
        self.label_map = assets['label_map']
        self.feature_params = assets['feature_params']
        
    def predict(self, data):
        window_samples = int(self.feature_params['window_length_sec'] * SAMPLE_RATE)
        if data.shape[1] < window_samples:
            return "rest"
            
        window = data[:, -window_samples:]
        features = self._extract_features(window)
        if features.size == 0:
            return "rest"
            
        features = self.scaler.transform(features)
        return self.label_map[self.model.predict(features)[0]]
        
    def _extract_features(self, X):
        all_features = []
        for window in X:
            window_features = []
            for ch_signal in window:
                window_features.extend(self.compute_features(ch_signal))
            all_features.append(window_features)
        return np.array(all_features)


# def real_time_loop():
#     state = HandControlState()
#     animator = HandAnimator("Adroit_hand.xml")
#     classifier = HandPoseClassifier("hand_model.joblib")
    
#     try:
#         board = init_cyton()
        
#         # Data collection thread
#         def collect_data():
#             while state.running:
#                 new_data = collect_fixed_samples(board, int(PREDICTION_INTERVAL * SAMPLE_RATE))
#                 with state.lock:
#                     state.data_buffer = np.concatenate([state.data_buffer, new_data], axis=1)[:, -int(BUFFER_DURATION * SAMPLE_RATE):]
#                 time.sleep(PREDICTION_INTERVAL)
                
#         # Control thread
#         def control_logic():
#             while state.running:
#                 if (time.time() - state.last_action_time) > ACTION_HOLD_DURATION and state.current_pose != "rest":
#                     animator.animate("rest")
#                     state.current_pose = "rest"
                    
#                 if state.current_pose == "rest":
#                     with state.lock:
#                         prediction = classifier.predict(state.data_buffer)
#                         if prediction != "rest":
#                             state.prediction_buffer.append(prediction)
#                             if len(state.prediction_buffer) > 3:
#                                 state.prediction_buffer.pop(0)
                                
#                             if all(p == prediction for p in state.prediction_buffer):
#                                 animator.animate(prediction)
#                                 state.current_pose = prediction
#                                 state.last_action_time = time.time()
#                 time.sleep(0.1)
        
#         # Start threads
#         threading.Thread(target=collect_data, daemon=True).start()
#         threading.Thread(target=control_logic, daemon=True).start()
        
#         # Rendering loop
#         while state.running:
#             animator.renderer.update_scene(animator.data)
#             media.show_video([animator.renderer.render()], fps=30)
#             if 'escape' in event.getKeys():
#                 state.running = False
                
#     finally:
#         board.stop_stream()
#         board.release_session()

# =======================
# Test System
# =======================
def test_system(X_test, y_test):
    state = HandControlState()
    animator = HandAnimator("Adroit/Adroit_hand.xml")
    classifier = HandPoseClassifier("hand_model.joblib")
    
    for _ in range(5):
        idx = np.random.randint(len(X_test))
        with state.lock:
            state.data_buffer = X_test[idx]
            
        pred = classifier.predict(state.data_buffer)
        print(f"True: {y_test[idx]}, Predicted: {pred}")
        
        if pred != "rest":
            animator.animate(pred)
            time.sleep(10)
            animator.animate("rest")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    
    # if args.test:
        # Load your test data
    test_system(X_test_scaled, y_test)
    # else:
    #     real_time_loop()