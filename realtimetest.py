# hand_control_system.py
import time
import numpy as np
import pickle
import mujoco
import mediapy as media

class HandAnimator:
    def __init__(self, model_path):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, 480, 640)
        
        # Create joint mapping directly from qpos structure
        self.joint_positions = {
            self.model.joint(i).name: i 
            for i in range(self.model.njnt)
        }
        self.qpos_size = self.data.qpos.size  # Get actual qpos size

    def animate_pose(self, command, duration=1.0):
        q_target = self._get_target_config(command)
        if q_target is None or len(q_target) != self.qpos_size:
            print(f"Configuration mismatch: Expected {self.qpos_size} joints, got {len(q_target) if q_target else 0}")
            return

        q_start = self.data.qpos.copy()
        start_time = time.time()
        
        while (time.time() - start_time) < duration:
            alpha = (time.time() - start_time) / duration
            self.data.qpos[:] = q_start + alpha * (q_target - q_start)
            mujoco.mj_forward(self.model, self.data)
            time.sleep(0.01)

    def _get_target_config(self, command):
        """Create configuration matching actual qpos size"""
        config = np.zeros_like(self.data.qpos)  # Match exact qpos dimensions
        
        try:
            if command == "rest":
                return config
            elif command.lower() == "fist":
                for name in self.joint_map:
                    if "FJ2" in name: config[self.joint_map[name]] = 1.6
                    elif "FJ1" in name: config[self.joint_map[name]] = 1.0
                    elif "THJ3" in name: config[self.joint_map[name]] = 1.3
                    elif "THJ0" in name: config[self.joint_map[name]] = -1.57
                    elif "THJ1" in name: config[self.joint_map[name]] = -0.52
                    elif "THJ2" in name: config[self.joint_map[name]] = 0.1
                return config
            elif command.lower() == "okay":
                for name in self.joint_map:
                    if "FFJ1" in name: config[self.joint_map[name]] = 1.6
                    elif "FFJ2" in name: config[self.joint_map[name]] = 1.0
                    elif "THJ4" in name: config[self.joint_map[name]] = 0.2
                    elif "THJ3" in name: config[self.joint_map[name]] = 1.0
                    elif "THJ0" in name: config[self.joint_map[name]] = -1.0
                    elif "THJ1" in name: config[self.joint_map[name]] = -0.075
                    elif "THJ2" in name: config[self.joint_map[name]] = 0.135
                return config
            elif command.lower() == "two":
                for name in self.joint_map:
                    if "FFJ3" in name: config[self.joint_map[name]] = 0.44
                    elif "MFJ3" in name: config[self.joint_map[name]] = -0.44
                    elif "RFJ1" in name: config[self.joint_map[name]] = 1.6
                    elif "RFJ2" in name: config[self.joint_map[name]] = 1.6
                    elif "LFJ2" in name: config[self.joint_map[name]] = 1.6
                    elif "LFJ1" in name: config[self.joint_map[name]] = 1.6
                    elif "THJ4" in name: config[self.joint_map[name]] = 1.0
                    elif "THJ3" in name: config[self.joint_map[name]] = 1.3
                    elif "THJ2" in name: config[self.joint_map[name]] = 0.3
                    elif "THJ1" in name: config[self.joint_map[name]] = -0.5
                    elif "THJ0" in name: config[self.joint_map[name]] = -1.57
                return config
            elif command.lower() == "flat":
                for name in self.joint_map:
                    if "FFJ3" in name: config[self.joint_map[name]] = 0.44
                    elif "MFJ3" in name: config[self.joint_map[name]] = 0.09
                    elif "RFJ3" in name: config[self.joint_map[name]] = -0.4
                    elif "LFJ3" in name: config[self.joint_map[name]] = -0.44
                return config
            return None
        except KeyError as e:
            print(f"Configuration error: {e}")
            return None

def test_system():
    # Load all components from your pickle file
    with open('hand_pose_classifier.pkl', 'rb') as f:
        saved_data = pickle.load(f)
    
    # Extract components exactly as you saved them
    pipeline = saved_data['pipeline']
    label_map = saved_data['label_map']
    X_test_scaled = saved_data['X_test']
    y_test = saved_data['y_test']
    
    # Initialize animator with your hand model
    animator = HandAnimator("Adroit/Adroit_hand.xml")
    
    # Get the SVM model from the pipeline
    svm_model = pipeline.steps[1][1]

    for i in range(5):  # Run 5 test cases
        idx = np.random.randint(len(X_test_scaled))
        sample = X_test_scaled[idx]

        
        # Make prediction using the pipeline's SVM directly
        prediction = svm_model.predict(sample.reshape(1, -1))[0]
        pred_label = str(prediction)

        print(f"\nTest {i+1}/5")
        print(f"Predicted: {pred_label}")
        
        if pred_label != "rest":
            animator.animate_pose(pred_label)
            time.sleep(2)
            animator.animate_pose("rest")
            time.sleep(1)

if __name__ == "__main__":
    test_system()