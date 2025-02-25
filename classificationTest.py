import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, DepthwiseConv2D, SeparableConv2D, AveragePooling2D, Dropout, Flatten, Dense, Activation
from tensorflow.keras.constraints import max_norm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import numpy as np
from collections import Counter

pkl_file = 'data\emg_handposes\sub-01\ses-01\emg_trial_data.pkl'
 
with open(pkl_file, 'rb') as f:
    trial_results=pickle.load(f)

fixed_length = 1052
for trial in trial_results:
    if trial["data"] is not None:
        if trial["data"].shape[1] > fixed_length:
            trial["data"] = trial["data"][:,:fixed_length]


pose_counts = Counter(trial["pose"] for trial in trial_results)

print("Pose Counts")
for pose, count in pose_counts.items():
    print(f"{pose}: {count}")

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


"""
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
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2, verbose=1)

# Evaluate on the test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print("Test accuracy: {:.2f}%".format(test_acc * 100))


"""