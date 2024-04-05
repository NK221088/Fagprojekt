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


# # Assuming X is your data
# samples, features, time = X.shape

# # Reshape to 2D
# X_2D = np.reshape(X, (samples*features, time))

# # Standardize
# scaler = StandardScaler()
# X_standardized_2D = scaler.fit_transform(X_2D)

# # Reshape back to 3D
# X = np.reshape(X_standardized_2D, (samples, features, time))

X = (X - np.mean(X, axis = 0)) / np.std(X, axis = 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = tf.convert_to_tensor(X_train)
y_train = tf.convert_to_tensor(y_train)
X_test = tf.convert_to_tensor(X_test)
y_test = tf.convert_to_tensor(y_test)

# Basic model:
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(np.shape(X)[1], np.shape(X)[2])),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(len(np.unique(y)))
])
# Output layer with no activation function (to ensure logits)
model.add(layers.Dense(len(np.unique(y))))

# Softmax activation applied separately to logits
model.add(layers.Activation('softmax'))

# Convolutional model:
model = models.Sequential()

# Convolutional base
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(np.shape(X)[1], np.shape(X)[2], 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))

# Dense layers on top
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))  # Dropout layer to reduce overfitting

# Output layer
model.add(layers.Dense(len(np.unique(y)), activation='softmax'))

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
class fNirs_LRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, decay_steps, decay_rate, staircase):
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.staircase = staircase

    def __call__(self, step):
        lr = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.initial_learning_rate,
            decay_steps=self.decay_steps,
            decay_rate=self.decay_rate,
            staircase=self.staircase)(step)
        return lr / (step + 1)

initial_learning_rate = 0.1
decay_steps = 10000
decay_rate = 0.9

optimizer = keras.optimizers.Adam(
    learning_rate=fNirs_LRSchedule(
        initial_learning_rate = initial_learning_rate,
        decay_steps = decay_steps,
        decay_rate = decay_rate,
        staircase = True
    ),
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    weight_decay=None,
    clipnorm=None,
    clipvalue=None,
    global_clipnorm=None,
    use_ema=False,
    ema_momentum=0.99,
    ema_overwrite_frequency=None,
    name="adam",
    **kwargs
)

model.compile(optimizer=optimizer,
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5)

model.evaluate(X_test,  y_test, verbose=2)