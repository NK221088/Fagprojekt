import tensorflow as tf
from load_data_function import load_data
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import TensorBoard
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from clr_callback import CyclicLR
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

def extract_features(X, model):
    # Build the model by making a dummy prediction
    dummy_data = tf.zeros((1, *X.shape[1:]))
    model.predict(dummy_data)
    
    # Create the feature extractor model
    feature_extractor = tf.keras.models.Model(inputs=model.input, outputs=model.layers[-2].output)
    
    # Extract features
    features = feature_extractor.predict(X)
    return features

def ANN_classifier(Xtrain, ytrain, Xtest, ytest, theta):
    # Allow memory growth for the GPU
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except RuntimeError as e:
            print(e)

    Xtest = (Xtest - np.mean(Xtrain, axis=0)) / np.std(Xtrain, axis=0)
    Xtrain = (Xtrain - np.mean(Xtrain, axis=0)) / np.std(Xtrain, axis=0)

    X_train = tf.convert_to_tensor(Xtrain)
    y_train = tf.convert_to_tensor(ytrain)
    y_train = tf.cast(y_train, tf.int32)
    X_test = tf.convert_to_tensor(Xtest)
    y_test = tf.convert_to_tensor(ytest)
    y_test = tf.cast(y_test, tf.int32)

    # Define your model
    model = tf.keras.models.Sequential()

    # Flatten input if necessary
    if theta.get("layer_type") == "dense":
        model.add(tf.keras.layers.Flatten(input_shape=(X_train.shape[1], X_train.shape[2])))

    # Define activation function
    def get_activation_function(name):
        if name == "leakyrelu":
            return tf.keras.layers.LeakyReLU()
        elif name == "elu":
            return tf.keras.layers.ELU()
        else:
            return tf.keras.activations.relu

    activation_function = get_activation_function(theta.get("activation_function", "relu"))

    # Retrieve dropout rate or set default
    dropout_rate = theta.get("dropout_rate", 0.2)
    initial_dropout_rate = dropout_rate * 0.5  # Scaling down for initial layers

    # Add initial layers based on the type specified in theta
    if theta.get("layer_type") == "conv1d":
        model.add(tf.keras.layers.Conv1D(filters=theta["neuron1"], kernel_size=3, activation=activation_function, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
        model.add(tf.keras.layers.Dropout(initial_dropout_rate))
    elif theta.get("layer_type") in ["lstm", "gru"]:
        recurrent_layer = tf.keras.layers.LSTM if theta["layer_type"] == "lstm" else tf.keras.layers.GRU
        model.add(recurrent_layer(theta["neuron1"], activation=activation_function, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(tf.keras.layers.Dropout(initial_dropout_rate))
    else:
        model.add(tf.keras.layers.Dense(theta["neuron1"], activation=activation_function))
        model.add(tf.keras.layers.Dropout(initial_dropout_rate))

    # Add additional layers based on theta["layers"]
    if theta["layers"] == 6:
        if theta.get("layer_type") == "conv1d":
            model.add(tf.keras.layers.Conv1D(filters=theta["neuron2"], kernel_size=3, activation=activation_function))
            model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
        elif theta.get("layer_type") in ["lstm", "gru"]:
            model.add(recurrent_layer(theta["neuron2"], activation=activation_function, return_sequences=True))
        else:
            model.add(tf.keras.layers.Dense(theta["neuron2"], activation=activation_function))
        model.add(tf.keras.layers.Dropout(dropout_rate))
    elif theta["layers"] == 8:
        if theta.get("layer_type") == "conv1d":
            model.add(tf.keras.layers.Conv1D(filters=theta["neuron2"], kernel_size=3, activation=activation_function))
            model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
            model.add(tf.keras.layers.Conv1D(filters=theta["neuron1"], kernel_size=3, activation=activation_function))
            model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
        elif theta.get("layer_type") in ["lstm", "gru"]:
            model.add(recurrent_layer(theta["neuron2"], activation=activation_function, return_sequences=True))
            model.add(recurrent_layer(theta["neuron1"], activation=activation_function, return_sequences=True))
        else:
            model.add(tf.keras.layers.Dense(theta["neuron2"], activation=activation_function))
            model.add(tf.keras.layers.Dropout(dropout_rate))
            model.add(tf.keras.layers.Dense(theta["neuron1"], activation=activation_function))
        model.add(tf.keras.layers.Dropout(dropout_rate))

    # Add the output layer for binary classification
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    # Compile the model
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    epochs = 300
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

    if theta["learning_rate"] == "decrease":
        initial_learning_rate = 0.001
        decay_steps = 50
        decay_rate = 0.9

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=fNirs_LRSchedule(
                initial_learning_rate=initial_learning_rate,
                decay_steps=decay_steps,
                decay_rate=decay_rate,
            )
        )
        model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[tensorboard_callback], verbose=0)
    elif theta["learning_rate"] == "clr":
        initial_learning_rate = 0.00025
        max_learning_rate = 0.001
        step_size = 20

        optimizer = tf.keras.optimizers.Adam()
        model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
        clr = CyclicLR(base_lr=initial_learning_rate, max_lr=max_learning_rate, step_size=step_size, mode='exp_range')
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[tensorboard_callback, clr], verbose=0)

    # Ensure the model is built
    dummy_data = tf.zeros((1, X_train.shape[1], X_train.shape[2]))
    model(dummy_data)  # This will build the model

    # Create the feature extractor model
    feature_extractor = tf.keras.models.Model(inputs=model.input, outputs=model.layers[-2].output)
    
    # Extract features from the trained network
    train_features = feature_extractor.predict(X_train)
    test_features = feature_extractor.predict(X_test)

    # Train an SVM on the extracted features
    svm = SVC(kernel='rbf')
    svm.fit(train_features, ytrain)

    # Evaluate the SVM
    y_pred = svm.predict(test_features)
    accuracy = accuracy_score(ytest, y_pred)

    # Clear session and delete model to free up memory
    tf.keras.backend.clear_session()
    del model
    gc.collect()

    return accuracy
