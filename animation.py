import os
import time
import mujoco
import mujoco.viewer
import numpy as np
import mediapy as media
import matplotlib.pyplot as plt
np.set_printoptions(precision=3, suppress=True, linewidth=100)

model = mujoco.MjModel.from_xml_path("/Users/jmalegaonkar/Desktop/EMGrip/Adroit/Adroit_hand.xml")
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