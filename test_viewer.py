import os
import time
import mujoco
import mujoco.viewer
import numpy as np
import mediapy as media
import matplotlib.pyplot as plt
from threading import Thread, Event
import pickle
from sklearn.preprocessing import StandardScaler
np.set_printoptions(precision=3, suppress=True, linewidth=100)

# Load the model
model = mujoco.MjModel.from_xml_path("/Users/jmalegaonkar/Desktop/EMGrip/Adroit/Adroit_hand.xml")
data = mujoco.MjData(model)
# Debug print to understand model structure
print(f"Number of actuators (controls): {model.nu}")
print("Actuator names:", [model.actuator(i).name for i in range(model.nu)])

# Create joint mapping
joint_map = {}
for i in range(model.nu):
    joint_id = model.actuator(i).trnid[0]
    joint_name = model.joint(joint_id).name
    joint_map[joint_name] = i

print("\nJoint mapping:")
for name, idx in joint_map.items():
    print(f"{name}: {idx}")

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

class HandSimulation:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.current_command = "rest"
        self.command_changed = Event()
        self.exit_flag = Event()
        
    def set_command(self, command):
        """Set a new command and notify the simulation thread"""
        if command.lower() == "exit":
            self.exit_flag.set()
            return
            
        self.current_command = command
        self.command_changed.set()
        print(f"Received command: {command}")
    
    def move_to_configuration(self, target_config, duration=2.0):
        """Smoothly move the hand to the target configuration over the specified duration"""
        # Store starting configuration
        start_config = self.data.qpos[:model.nu].copy()
        
        # Animation parameters
        start_time = time.time()
        
        while time.time() - start_time < duration:
            # Interpolation factor (0 to 1)
            alpha = (time.time() - start_time) / duration
            alpha = min(alpha, 1.0)  # Clamp to 1.0
            
            # Interpolate between start and desired positions
            self.data.qpos[:model.nu] = start_config + alpha * (target_config - start_config)
            
            # Update model state
            mujoco.mj_forward(self.model, self.data)
            
            # Give control back to the viewer
            time.sleep(0.01)
    
    def run_simulation(self):
        """Main simulation loop"""
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            # Initial reset
            mujoco.mj_resetData(self.model, self.data)
            
            # Set to rest position initially
            rest_config = get_desired_configuration("rest")
            self.data.qpos[:model.nu] = rest_config
            mujoco.mj_forward(self.model, self.data)
            
            # Main loop
            while not self.exit_flag.is_set():
                # Update the viewer
                viewer.sync()
                
                # Wait for a command change or timeout (to keep updating viewer)
                if self.command_changed.wait(timeout=0.1):
                    self.command_changed.clear()
                    
                    # Skip if the command is "rest" and we're already at rest
                    if self.current_command.lower() == "rest" and np.allclose(self.data.qpos[:model.nu], rest_config, atol=1e-3):
                        continue
                    
                    # Get the desired configuration for the command
                    desired_config = get_desired_configuration(self.current_command)
                    
                    if desired_config is not None:
                        # Move to the desired configuration
                        print(f"Moving to {self.current_command} position...")
                        self.move_to_configuration(desired_config)
                        
                        # Hold the position for 10 seconds (if not "rest")
                        if self.current_command.lower() != "rest":
                            print(f"Holding {self.current_command} position for 10 seconds...")
                            hold_start = time.time()
                            
                            while time.time() - hold_start < 10.0 and not self.command_changed.is_set():
                                viewer.sync()
                                time.sleep(0.1)
                            
                            # If no new command came in during the hold, go back to rest
                            if not self.command_changed.is_set():
                                print("Returning to rest position...")
                                self.move_to_configuration(rest_config)
                    else:
                        print(f"Invalid command: {self.current_command}")

def run_hand_simulation():
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

def load_model(model_path='hand_pose_classifier.pkl'):
    """Load the trained classifier model"""
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        print("Model loaded successfully")
        return model_data
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def run_with_ml_predictions(model_path='hand_pose_classifier.pkl', continuous=True):
    """Run the hand simulation with predictions from the ML model"""
    # Load the model
    model_data = load_model(model_path)
    if model_data is None:
        print("Failed to load model. Exiting.")
        return
    
    # Extract components
    pipeline = model_data['pipeline']
    label_map = model_data['label_map']
    
    # Reverse the label mapping to convert numeric predictions to gesture names
    reverse_label_map = {v: k for k, v in label_map.items()}
    
    # Create and start the simulation
    simulation = HandSimulation(model, data)
    simulation_thread = Thread(target=simulation.run_simulation)
    simulation_thread.daemon = True
    simulation_thread.start()
    
    print("\nHand simulation started with ML model predictions")
    print("Press Ctrl+C to exit")
    
    try:
        if continuous:
            # Example of continuous prediction (you would replace this with your actual input)
            # This is where you would integrate your actual EMG data acquisition
            test_data = model_data['X_test']  # Using saved test data for demonstration
            
            for i, test_sample in enumerate(test_data[:10]):  # Just using 10 samples for demo
                # Make prediction
                prediction = pipeline.predict(test_sample.reshape(1, -1))[0]
                gesture_name = reverse_label_map.get(prediction, "unknown")
                
                print(f"Prediction {i+1}: {gesture_name} (class {prediction})")
                
                # Execute the predicted gesture
                simulation.set_command(gesture_name)
                
                # Wait for the gesture to complete (movement + hold + return to rest)
                # 2s movement + 10s hold + 2s return + 1s buffer = 15s
                time.sleep(15)
        else:
            # Single prediction example
            test_index = 0  # Change this to the index of the test sample you want to use
            test_sample = model_data['X_test'][test_index]
            
            # Make prediction
            prediction = pipeline.predict(test_sample.reshape(1, -1))[0]
            gesture_name = reverse_label_map.get(prediction, "unknown")
            
            print(f"Predicted gesture: {gesture_name} (class {prediction})")
            
            # Execute the predicted gesture
            simulation.set_command(gesture_name)
            
            # Wait for completion
            time.sleep(15)
        
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        # Clean up
        simulation.set_command("exit")
        simulation_thread.join(timeout=1.0)
        print("Simulation ended")

def run_real_time_predictions(model_path='hand_pose_classifier.pkl'):
    """
    This function would integrate with your EMG acquisition system
    to get real-time predictions and control the hand.
    """
    # Load the model
    model_data = load_model(model_path)
    if model_data is None:
        print("Failed to load model. Exiting.")
        return
    
    # Extract components
    pipeline = model_data['pipeline']
    label_map = model_data['label_map']
    feature_params = model_data['feature_params']
    
    # Reverse the label mapping
    reverse_label_map = {v: k for k, v in label_map.items()}
    
    # Create and start the simulation
    simulation = HandSimulation(model, data)
    simulation_thread = Thread(target=simulation.run_simulation)
    simulation_thread.daemon = True
    simulation_thread.start()
    
    print("\nHand simulation started with real-time EMG predictions")
    print("Press Ctrl+C to exit")
    
    # Here you would add code to:
    # 1. Initialize your EMG data acquisition system
    # 2. Set up a loop to continuously capture EMG data
    # 3. Process the EMG data into the same feature format used during training
    # 4. Make predictions using the pipeline
    # 5. Send commands to the simulation
    
    try:
        # Placeholder for your real-time EMG processing code
        print("Waiting for EMG input...")
        
        # Example of how the loop would work (pseudo-code):
        """
        while True:
            # Get EMG data
            emg_data = get_emg_data()
            
            # Extract features
            features = extract_features(emg_data, feature_params)
            
            # Make prediction
            prediction = pipeline.predict(features.reshape(1, -1))[0]
            gesture_name = reverse_label_map.get(prediction, "unknown")
            
            # Execute the predicted gesture if it's changed
            if gesture_name != current_gesture:
                current_gesture = gesture_name
                simulation.set_command(gesture_name)
            
            # Wait for the next EMG sample
            time.sleep(0.1)
        """
        
        # For demo purposes, just wait for Ctrl+C
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        # Clean up
        simulation.set_command("exit")
        simulation_thread.join(timeout=1.0)
        print("Simulation ended")

# Main entry point
if __name__ == "__main__":    
    # 1. For manually input commands:
    run_hand_simulation()
    
    # 2. For testing with saved test data from the ML model:
    # run_with_ml_predictions(continuous=True)
    
    # 3. For real-time EMG predictions:
    # run_real_time_predictions()