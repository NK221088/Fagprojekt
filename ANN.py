import tensorflow as tf
from load_data_function import load_data
from fNirs_processesing_fNirs_motor import data_fNirs_motor
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



# Allow memory growth for the GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        # Set memory growth to avoid allocating all GPU memory at once
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except RuntimeError as e:
        print(e)
        
data_set = "fNirs_motor_full_data"
epoch_type = "Tapping"
short_channel_correction = True
negative_correlation_enhancement = True

all_epochs, data_name, all_data, freq = load_data(data_set = data_set, short_channel_correction = short_channel_correction, negative_correlation_enhancement = negative_correlation_enhancement)

X = np.concatenate((all_data[epoch_type],all_data["Control"]), axis = 0)
y = np.concatenate((np.ones(len(all_data[epoch_type])), np.zeros(len(all_data["Control"]))), axis=0)


# Assuming X is your data
samples, features, time = X.shape

# Reshape to 2D
X_2D = np.reshape(X, (samples*features, time))

# Standardize
scaler = StandardScaler()
X_standardized_2D = scaler.fit_transform(X_2D)

# Reshape back to 3D
X = np.reshape(X_standardized_2D, (samples, features, time))

# X = (X - np.mean(X, axis = 0)) / np.std(X, axis = 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)
X_train = tf.convert_to_tensor(X_train)
y_train = tf.convert_to_tensor(y_train)
X_test = tf.convert_to_tensor(X_test)
y_test = tf.convert_to_tensor(y_test)

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(np.shape(X)[1], np.shape(X)[2])),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(len(np.unique(y)))
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5)

model.evaluate(X_test,  y_test, verbose=2)