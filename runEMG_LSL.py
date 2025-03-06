import pygame
import time
import os
import numpy as np
import pickle
from pylsl import StreamInlet, resolve_byprop

# Initialize pygame
pygame.init()

# Set up display
WIDTH, HEIGHT = 800, 600
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Hand Pose Experiment")

# Colors
WHITE = (255, 255, 255)
GREY = (169, 169, 169)
FONT = pygame.font.Font(None, 80)

# Experiment Settings
subject = 1
session = 1
SAMPLE_RATE = 250  # Assumed from OpenBCI
reaction_time = 0  # seconds to discard initially (calibration)
discard_samples = int(reaction_time * SAMPLE_RATE)

n_rounds_per_pose = 1
n_cycles = 1
rest_duration_seconds = 3
trial_duration_seconds = 3
rest_samples = int(rest_duration_seconds * SAMPLE_RATE)
trial_samples = int(trial_duration_seconds * SAMPLE_RATE)

save_dir = f'data/emg_handposes/sub-{subject:02d}/ses-{session:02d}/'
os.makedirs(save_dir, exist_ok=True)
save_file = os.path.join(save_dir, 'emg_trial_data.pkl')

# Load hand pose images
hand_poses = ['fist', 'flat', 'okay', 'two']
image_stimuli = {pose: pygame.image.load(os.path.join("positions", f"{pose}.jpg")) for pose in hand_poses}

# Find LSL EEG Stream
print("Looking for an EEG LSL stream...")
streams = resolve_byprop('name', 'test', timeout=5)
if not streams:
    print("No LSL stream found. Exiting...")
    exit(1)

inlet = StreamInlet(streams[0])

def collect_fixed_samples(num_samples):
    """Collect 'num_samples' from the LSL stream and return (n_channels, num_samples)."""
    collected = []
    
    while len(collected) < num_samples:
        # Check for quit or escape key
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                global running
                running = False
                return None  # Return None to indicate early exit
        
        # Pull EEG data
        chunk, _ = inlet.pull_chunk()
        if chunk:
            collected.extend(chunk)
    
    return np.array(collected).T[:, :num_samples]  # Ensure correct shape (channels, samples)


def display_text(text, duration):
    """Display text for a given duration in seconds."""
    win.fill(GREY)
    text_surface = FONT.render(text, True, WHITE)
    text_rect = text_surface.get_rect(center=(WIDTH // 2, HEIGHT // 2))
    win.blit(text_surface, text_rect)
    pygame.display.update()
    time.sleep(duration)

def display_image(image, duration):
    """Display an image for a given duration in seconds."""
    win.fill(GREY)
    win.blit(image, (WIDTH // 10, HEIGHT // 10))
    pygame.display.update()
    time.sleep(duration)

# Main Experiment Loop
# Main Experiment Loop
trial_results = []
running = True

try:
    for pose in hand_poses:
        for rnd in range(1, n_rounds_per_pose + 1):
            for cycle in range(1, n_cycles + 1):
                if not running:
                    break  # Exit inner loop

                # Rest Period
                display_text("Rest", rest_duration_seconds)
                raw_rest = collect_fixed_samples(rest_samples)
                if raw_rest is None:  # Exit if early termination detected
                    break
                trial_results.append({
                    "pose": "Rest",
                    "round": rnd,
                    "cycle": cycle,
                    "samples": rest_samples - discard_samples,
                    "data": raw_rest[:, discard_samples:]
                })

                if not running:
                    break  # Exit inner loop

                # Pose Period
                display_image(image_stimuli[pose], trial_duration_seconds)
                raw_pose = collect_fixed_samples(trial_samples)
                if raw_pose is None:  # Exit if early termination detected
                    break
                trial_results.append({
                    "pose": pose,
                    "round": rnd,
                    "cycle": cycle,
                    "samples": trial_samples - discard_samples,
                    "data": raw_pose[:, discard_samples:]
                })

                if not running:
                    break  # Exit inner loop
            if not running:
                break  # Exit middle loop
        if not running:
            break  # Exit outer loop

except KeyboardInterrupt:
    print("Experiment interrupted manually.")

pygame.quit()

# Save Data if not interrupted
if running:
    with open(save_file, 'wb') as f:
        pickle.dump(trial_results, f)
    print("Experiment finished. Data saved to:", save_file)
else:
    print("Experiment terminated early. No data saved.")

