import os
import time
import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
from threading import Thread, Event
import pickle
from sklearn.preprocessing import StandardScaler
from pylsl import StreamInlet, resolve_byprop
np.set_printoptions(precision=3, suppress=True, linewidth=100)

SAMPLE_RATE = 250
selected_channels = [0, 3, 6]
window_length_sec = 0.2
window_length = int(window_length_sec * SAMPLE_RATE)


model = mujoco.MjModel.from_xml_path("/Users/jmalegaonkar/Desktop/EMGrip/Adroit/Adroit_hand.xml")
data = mujoco.MjData(model)
# print(f"Number of actuators (controls): {model.nu}")
# print("Actuator names:", [model.actuator(i).name for i in range(model.nu)])

# Create joint mapping
joint_map = {}
for i in range(model.nu):
    joint_id = model.actuator(i).trnid[0]
    joint_name = model.joint(joint_id).name
    joint_map[joint_name] = i

# print("\nJoint mapping:")
# for name, idx in joint_map.items():
#     print(f"{name}: {idx}")

def get_desired_configuration(command):
    """Get the joint configuration for a specific hand pose command"""
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

class HandSimulation:
    """Class to handle the hand simulation and movement logic"""
    def __init__(self, model, data, hold_time=10.0):
        self.model = model
        self.data = data
        self.current_command = "rest"
        self.command_changed = Event()
        self.exit_flag = Event()
        self.hold_time = hold_time
        self.last_command_time = 0
        self.in_rest_position = True
        
    def set_command(self, command):
        """Set a new command and notify the simulation thread"""
        if command.lower() == "exit":
            self.exit_flag.set()
            return
            
        # Only update if it's a new command or we've returned to rest since the last command
        current_time = time.time()
        if (command.lower() != self.current_command.lower() or 
            (self.in_rest_position and current_time - self.last_command_time > 1.0)):
            
            self.current_command = command
            self.last_command_time = current_time
            self.command_changed.set()
            print(f"Executing command: {command}")
    
    def move_to_configuration(self, target_config, duration=2.0):
        """Smoothly move the hand to the target configuration over the specified duration"""
        # Store starting configuration
        start_config = self.data.qpos[:model.nu].copy()
        
        # Animation parameters
        start_time = time.time()
        
        while time.time() - start_time < duration and not self.command_changed.is_set():
            # Interpolation factor (0 to 1)
            alpha = (time.time() - start_time) / duration
            alpha = min(alpha, 1.0)  # Clamp to 1.0
            
            # Interpolate between start and desired positions
            self.data.qpos[:model.nu] = start_config + alpha * (target_config - start_config)
            
            # Update model state
            mujoco.mj_forward(self.model, self.data)
            
            # Give control back to the viewer
            time.sleep(0.01)
        
        # Update the rest position flag
        rest_config = get_desired_configuration("rest")
        self.in_rest_position = np.allclose(self.data.qpos[:model.nu], rest_config, atol=1e-2)
    
    def run_simulation(self):
        """Main simulation loop"""
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            # Initial reset
            mujoco.mj_resetData(self.model, self.data)
            
            # Set to rest position initially
            rest_config = get_desired_configuration("rest")
            self.data.qpos[:model.nu] = rest_config
            mujoco.mj_forward(self.model, self.data)
            self.in_rest_position = True
            
            # Main loop
            while not self.exit_flag.is_set():
                # Update the viewer
                viewer.sync()
                
                # Wait for a command change or timeout (to keep updating viewer)
                if self.command_changed.wait(timeout=0.1):
                    self.command_changed.clear()
                    
                    # Skip if the command is "rest" and we're already at rest
                    if self.current_command.lower() == "rest" and self.in_rest_position:
                        continue
                    
                    # Get the desired configuration for the command
                    desired_config = get_desired_configuration(self.current_command)
                    
                    if desired_config is not None:
                        # Move to the desired configuration
                        print(f"Moving to {self.current_command} position...")
                        self.move_to_configuration(desired_config)
                        self.in_rest_position = (self.current_command.lower() == "rest")
                        
                        # Hold the position for the specified time (if not "rest")
                        if self.current_command.lower() != "rest":
                            print(f"Holding {self.current_command} position for {self.hold_time} seconds...")
                            hold_start = time.time()
                            
                            while (time.time() - hold_start < self.hold_time and 
                                   not self.command_changed.is_set()):
                                viewer.sync()
                                time.sleep(0.1)
                            
                            # If no new command came in during the hold, go back to rest
                            if not self.command_changed.is_set():
                                print("Returning to rest position...")
                                self.move_to_configuration(rest_config)
                                self.in_rest_position = True
                    else:
                        print(f"Invalid command: {self.current_command}")

def compute_features(signal):
    """Extract features from an EMG signal channel"""
    mav = np.mean(np.abs(signal))
    ssc_count = sum(1 for i in range(1, len(signal) - 1)
                    if (signal[i] - signal[i-1]) * (signal[i+1] - signal[i]) < 0)
    zc_count = sum(1 for i in range(len(signal) - 1)
                   if (signal[i] > 0 and signal[i+1] < 0) or (signal[i] < 0 and signal[i+1] > 0))
    wl = np.sum(np.abs(np.diff(signal)))
    return np.array([mav, ssc_count, zc_count, wl])

def collect_fixed_samples(inlet, num_samples):
    """Collects 'num_samples' from the LSL stream and returns (n_channels, num_samples)."""
    collected = []
    while len(collected) < num_samples:
        chunk, _ = inlet.pull_chunk()
        if chunk:
            collected.extend(chunk)
    return np.array(collected).T[:, :num_samples]

def load_model(model_path='trained_model.pkl'):
    """Load the trained classifier model"""
    try:
        with open(model_path, 'rb') as f:
            model, scaler = pickle.load(f)
        
        print("Model loaded successfully")
        return model, scaler
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def load_hand_pose_model(model_path='hand_pose_classifier.pkl'):
    """Load the comprehensive hand pose classifier model"""
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        print("Hand pose classifier model loaded successfully")
        return model_data
    except Exception as e:
        print(f"Error loading hand pose model: {e}")
        return None

def run_emg_controlled_simulation(model_path='trained_model.pkl', hold_time=10.0):
    """Run the hand simulation with real-time EMG input"""
    # Load the ML model
    model, scaler = load_model(model_path)
    if model is None or scaler is None:
        print("Failed to load model. Exiting.")
        return
    
    # Label mapping (update this if your model uses different labels)
    label_map = {0: "fist", 1: "flat", 2: "okay", 3: "two", 4: "rest"}
    
    # Resolve LSL Stream
    print("Looking for an EEG/EMG LSL stream...")
    streams = resolve_byprop('name', 'test', timeout=5)
    if not streams:
        print("No LSL stream found. Exiting...")
        return
    
    inlet = StreamInlet(streams[0])
    print(f"Connected to stream: {streams[0].name()}")
    
    # Create and start the simulation
    simulation = HandSimulation(model, data, hold_time=hold_time)
    simulation_thread = Thread(target=simulation.run_simulation)
    simulation_thread.daemon = True
    simulation_thread.start()
    
    print("\nHand simulation started with real-time EMG control")
    print("Press Ctrl+C to exit")
    
    last_prediction = "rest"
    prediction_count = {}  # For tracking stable predictions
    
    try:
        while True:
            # Collect real-time EMG data
            emg_data = collect_fixed_samples(inlet, window_length)
            emg_data = emg_data[selected_channels, :]  # Select only the channels we use
            
            # Extract features from each channel
            features = []
            for ch_signal in emg_data:
                features.extend(compute_features(ch_signal))
            features = np.array(features).reshape(1, -1)
            
            # Standardize features
            features_scaled = scaler.transform(features)
            
            # Predict pose
            pose_prediction = model.predict(features_scaled)[0]
            predicted_pose = label_map[pose_prediction]
            
            # Simple stability filter - only accept predictions that are stable for a while
            if predicted_pose not in prediction_count:
                prediction_count = {pose: 0 for pose in label_map.values()}
                prediction_count[predicted_pose] = 1
            else:
                prediction_count[predicted_pose] += 1
                
                # If we have enough consistent predictions, execute the command
                if prediction_count[predicted_pose] >= 3 and predicted_pose != last_prediction:
                    print(f"Stable prediction: {predicted_pose}")
                    simulation.set_command(predicted_pose)
                    last_prediction = predicted_pose
                    prediction_count = {pose: 0 for pose in label_map.values()}
            
            # Small sleep to prevent overwhelming the CPU
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        # Clean up
        simulation.set_command("exit")
        simulation_thread.join(timeout=1.0)
        print("Simulation ended")

def run_interactive_hand_simulation():
    """Run the hand simulation with interactive command input"""
    # Create and start the simulation
    simulation = HandSimulation(model, data)
    simulation_thread = Thread(target=simulation.run_simulation)
    simulation_thread.daemon = True
    simulation_thread.start()
    
    print("\nHand simulation started. Enter commands to control the hand:")
    print("Available commands: rest, fist, okay, two, flat, exit")
    
    # Command input loop
    while True:
        command = input("> ")
        if command.lower() == "exit":
            simulation.set_command("exit")
            break
        simulation.set_command(command)
    
    # Wait for the simulation to clean up
    simulation_thread.join(timeout=1.0)
    print("Simulation ended")

def test_with_saved_model_data(model_path='hand_pose_classifier.pkl'):
    """Test the hand simulation with data from the saved model"""
    # Load the comprehensive model data
    model_data = load_hand_pose_model(model_path)
    if model_data is None:
        print("Failed to load model data. Exiting.")
        return
    
    # Extract test data and pipeline
    pipeline = model_data['pipeline']
    X_test = model_data['X_test']
    y_test = model_data['y_test']
    label_map = model_data['label_map']
    reverse_label_map = {v: k for k, v in label_map.items()}
    
    # Create and start the simulation
    simulation = HandSimulation(model, data)
    simulation_thread = Thread(target=simulation.run_simulation)
    simulation_thread.daemon = True
    simulation_thread.start()
    
    print("\nTesting with saved model data")
    
    try:
        # Test with a few samples
        for i in range(min(10, len(X_test))):
            # Get test sample
            test_sample = X_test[i].reshape(1, -1)
            
            # Make prediction
            prediction = pipeline.predict(test_sample)[0]
            gesture_name = reverse_label_map.get(prediction, "unknown")
            true_gesture = reverse_label_map.get(y_test[i], "unknown")
            
            print(f"Sample {i+1}: Predicted {gesture_name}, True {true_gesture}")
            
            # Execute the gesture
            simulation.set_command(gesture_name)
            
            # Wait for the gesture to complete
            time.sleep(15)  # Movement + hold + return to rest
            
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        # Clean up
        simulation.set_command("exit")
        simulation_thread.join(timeout=1.0)
        print("Simulation ended")

if __name__ == "__main__":
    
    # Real-time EMG control
    #run_emg_controlled_simulation(hold_time=10.0)
    
    # Interactive manual control
    run_interactive_hand_simulation()
    