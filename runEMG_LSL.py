import pygame
import time
import os
import numpy as np
import pickle
from pylsl import StreamInlet, resolve_byprop
import random
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


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
session = 3
SAMPLE_RATE = 250  # Assumed from OpenBCI
reaction_time = 0  # seconds to discard initially (calibration)
discard_samples = int(reaction_time * SAMPLE_RATE)

n_rounds_per_pose = 1
n_cycles = 2
rest_duration_seconds = 5
trial_duration_seconds = 5
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

def display_and_collect(stimulus, num_samples, is_image=False):
    """Displays a stimulus (text or image) while simultaneously collecting EEG data."""
    collected = []
    start_time = time.time()

    while len(collected) < num_samples:
        # Handle quit or escape key
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                global running
                running = False
                return None  # Stop experiment

        # Draw stimulus
        win.fill(GREY)

        if is_image:
            # Scale and center image
            img_width, img_height = stimulus.get_size()
            max_width, max_height = WIDTH * 0.8, HEIGHT * 0.8
            scale_factor = min(max_width / img_width, max_height / img_height)
            new_size = (int(img_width * scale_factor), int(img_height * scale_factor))
            stimulus_scaled = pygame.transform.smoothscale(stimulus, new_size)
            img_x = (WIDTH - new_size[0]) // 2
            img_y = (HEIGHT - new_size[1]) // 2
            win.blit(stimulus_scaled, (img_x, img_y))
        else:
            # Render text
            text_surface = FONT.render(stimulus, True, WHITE)
            text_rect = text_surface.get_rect(center=(WIDTH // 2, HEIGHT // 2))
            win.blit(text_surface, text_rect)

        pygame.display.update()

        # Pull EEG data while the stimulus is displayed
        chunk, _ = inlet.pull_chunk()
        if chunk:
            collected.extend(chunk)

    return np.array(collected).T[:, :num_samples]  # Ensure correct shape (channels, samples)



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
                raw_rest = display_and_collect("Rest", rest_samples, is_image=False)
                if raw_rest is None:  # Stop early if user quits
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
                raw_pose = display_and_collect(image_stimuli[pose], trial_samples, is_image=True)
                if raw_pose is None:  # Stop early if user quits
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



#Model Training

selected_channels = [0, 3, 6]
window_length_sec = 0.2


data_by_label = {}
for trial in trial_results:
    label = trial["pose"]
    if trial["data"] is not None and trial["data"].size > 0:
        data_by_label.setdefault(label, []).append(trial["data"])

for label, trials in data_by_label.items():
    for i in range(len(trials)):
        trial_data = trials[i]
        trials[i] = trial_data[selected_channels, :]

window_length = int(window_length_sec* SAMPLE_RATE)
overlap = 0
step = max(1, int(window_length * (1 - overlap)))  # step size

X_windowed, y_windowed = [], []
for label, trials in data_by_label.items():
    for trial in trials:
        if trial.size > 0:
            n_samples = trial.shape[1]
            for start in range(0, n_samples - window_length + 1, step):
                X_windowed.append(trial[:, start:start + window_length])
                y_windowed.append(label)

label_map = {"fist": 0, "flat": 1, "okay": 2, "two": 3, "Rest": 4}
y_windowed_int = [label_map[lbl] for lbl in y_windowed]
X_final = np.array(X_windowed)
y_final = np.array(y_windowed_int)




target_label = 4  # The class you want to downsample (e.g. 'rest')

from collections import defaultdict
class_data = defaultdict(list)
for x, label in zip(X_final, y_final):
    class_data[label].append(x)

min_other_class_size = min(len(samples)
                           for lbl, samples in class_data.items()
                           if lbl != target_label)

X_balanced = []
y_balanced = []

for lbl, samples in class_data.items():
    if lbl == target_label:
        if len(samples) > min_other_class_size:
            chosen = random.sample(samples, min_other_class_size)
        else:
            chosen = samples
        X_balanced.extend(chosen)
        y_balanced.extend([lbl] * len(chosen))
    else:
        # Keep every sample from other classes
        X_balanced.extend(samples)
        y_balanced.extend([lbl] * len(samples))

X_final = np.array(X_balanced)
y_final = np.array(y_balanced)

def compute_features(signal):
    mav = np.mean(np.abs(signal))
    ssc_count = sum(1 for i in range(1, len(signal) - 1)
                    if (signal[i] - signal[i-1]) * (signal[i+1] - signal[i]) < 0)
    zc_count = sum(1 for i in range(len(signal) - 1)
                   if (signal[i] > 0 and signal[i+1] < 0) or (signal[i] < 0 and signal[i+1] > 0))
    wl = np.sum(np.abs(np.diff(signal)))
    return np.array([mav, ssc_count, zc_count, wl])

all_features = []
for window in X_final:  # window.shape = (n_channels, window_length)
    window_features = []
    for ch_signal in window:
        window_features.extend(compute_features(ch_signal))
    all_features.append(window_features)
all_features = np.array(all_features)

excluded_labels = {}
keep_indices = [i for i, lbl in enumerate(y_final) if lbl not in excluded_labels]
features_filtered = all_features[keep_indices]
y_filtered = y_final[keep_indices]

X_train, X_test, y_train, y_test = train_test_split(features_filtered, y_filtered,
                                                    test_size=0.2, random_state=44)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

label_map = {0: "fist", 1: "flat", 2: "okay", 3: "two", 4:"rest"}
class_names = [label_map[i] for i in sorted(label_map.keys())]

param_grid_svm = {
    'C': [0.01, 0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1],
    'kernel': ['rbf', 'linear']
}

svm_clf = SVC()
grid_search = GridSearchCV(
    estimator=svm_clf,
    param_grid=param_grid_svm,
    scoring='accuracy',
    cv=5,
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train_scaled, y_train)

best_svm = grid_search.best_estimator_
print("Best Hyperparameters:", grid_search.best_params_)
print("Best CV Accuracy:", grid_search.best_score_)


with open('trained_model.pkl', 'wb') as f:
    pickle.dump((best_svm, scaler), f)

with open('trained_model.pkl', 'rb') as f:
    loaded_model, loaded_scaler = pickle.load(f)

print("Loaded Model:", loaded_model)
print("Loaded Scaler:", loaded_scaler)



