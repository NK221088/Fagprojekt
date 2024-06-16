import tensorflow as tf
from load_data_function import load_data
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import TensorBoard
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import gc
import tensorflow as tf
import os as os
import shutil

class fNirs_LRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, decay_steps, decay_rate):
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate

    def __call__(self, step):
        step = tf.cast(step, tf.float32)  # Cast step to tf.float32
        lr = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.initial_learning_rate,
            decay_steps=self.decay_steps,
            decay_rate=self.decay_rate,
            staircase=True)(step)
        return lr / (step + 1)
    
    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_steps": self.decay_steps,
            "decay_rate": self.decay_rate,
        }

def ANN_classifier(Xtrain, ytrain, Xtest, ytest, theta):
    # Allow memory growth for the GPU
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            # Set memory growth to avoid allocating all GPU memory at once
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except RuntimeError as e:
            print(e)
    
    Xtest = (Xtest - np.mean(Xtrain, axis = 0)) / np.std(Xtrain, axis = 0)
    Xtrain = (Xtrain - np.mean(Xtrain, axis = 0)) / np.std(Xtrain, axis = 0)
    
    X_train = tf.convert_to_tensor(Xtrain)
    y_train = tf.convert_to_tensor(ytrain)
    y_train = tf.cast(y_train, tf.int32)
    X_test = tf.convert_to_tensor(Xtest)
    y_test = tf.convert_to_tensor(ytest)
    y_test = tf.cast(y_test, tf.int32)
    
    # Define your model
    model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(np.shape(X_train)[1], np.shape(X_train)[2])),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer for binary classification. The units is 1, as the output of the sigmoid function represents the probability of belonging to the positive class
    ])
    
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    epochs = 100
    batch_size = 100
    log_dir = "logs/"
    # Clear old logs
    for file in os.listdir(log_dir):
        file_path = os.path.join(log_dir, file)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    initial_learning_rate = 0.01
    decay_steps = 50
    decay_rate = 0.9
    
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=fNirs_LRSchedule(
            initial_learning_rate = initial_learning_rate,
            decay_steps = decay_steps,
            decay_rate = decay_rate,
        )
    )
    model.compile(optimizer=optimizer,
                loss=loss_fn, 
                metrics=['accuracy'])
    
    model.fit(X_train, y_train, epochs = epochs, batch_size = batch_size, callbacks=[tensorboard_callback],verbose = 0)
    
    loss, accuracy = model.evaluate(X_test,  y_test, verbose=0)

    # Clear session and delete model to free up memory
    tf.keras.backend.clear_session()
    del model
    gc.collect()

    return accuracy