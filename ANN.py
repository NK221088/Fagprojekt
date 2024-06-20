import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix
import gc
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from clr_callback import CyclicLR  # Assuming you have a CyclicLR implementation
import seaborn as sns

def load_pretrained_weights(model, weights_path):
    try:
        pretrained_model = tf.keras.models.load_model(weights_path, compile=False)
        pretrained_weights = pretrained_model.layers[3].get_weights()
        model.layers[3].set_weights(pretrained_weights)
    except Exception as e:
        pass
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from clr_callback import CyclicLR
import gc
import os
import shutil

class fNirs_LRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, decay_steps, decay_rate):
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        lr = ExponentialDecay(
            initial_learning_rate=self.initial_learning_rate,
            decay_steps=self.decay_steps,
            decay_rate=self.decay_rate,
            staircase=True)(step)
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
            # Set memory growth to avoid allocating all GPU memory at once
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except RuntimeError as e:
            pass  # Handle exception if needed
    
    Xtest = (Xtest - np.mean(Xtrain, axis=0)) / np.std(Xtrain, axis=0)
    Xtrain = (Xtrain - np.mean(Xtrain, axis=0)) / np.std(Xtrain, axis=0)
    
    X_train = tf.convert_to_tensor(Xtrain)
    y_train = tf.convert_to_tensor(ytrain)
    y_train = tf.cast(y_train, tf.int32)
    X_test = tf.convert_to_tensor(Xtest)
    y_test = tf.convert_to_tensor(ytest)
    y_test = tf.cast(y_test, tf.int32)
    
    if theta["model"] == 1:
        # Define your model
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Flatten(input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(tf.keras.layers.Dense(theta["neuron1"], activation='relu'))
        model.add(tf.keras.layers.Dropout(0.2))
        
        if theta["layers"] == 6:
            model.add(tf.keras.layers.Dense(theta["neuron2"], activation='relu'))
            model.add(tf.keras.layers.Dropout(0.2))
        
        elif theta["layers"] == 8:
            model.add(tf.keras.layers.Dense(theta["neuron2"], activation='relu'))
            model.add(tf.keras.layers.Dropout(0.2))
            model.add(tf.keras.layers.Dense(theta["neuron1"], activation='relu'))
            model.add(tf.keras.layers.Dropout(0.2))
        
        # Add the output layer for binary classification
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        
        loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        epochs = 300
        batch_size = 100

        if theta["learning_rate"] == "decrease":
            initial_learning_rate = 0.001
            decay_steps = tf.constant(50, dtype=tf.int64)
            decay_rate = 0.9
            
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=fNirs_LRSchedule(
                    initial_learning_rate=initial_learning_rate,
                    decay_steps=decay_steps,
                    decay_rate=decay_rate,
                )
            )
            model.compile(optimizer=optimizer,
                        loss=loss_fn, 
                        metrics=['accuracy'])
            
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
            
        elif theta["learning_rate"] == "clr":
            initial_learning_rate = 0.0001
            max_learning_rate = 0.001
            step_size = 20
            
            optimizer = tf.keras.optimizers.Adam()
        
            model.compile(optimizer=optimizer,
                        loss=loss_fn,
                        metrics=['accuracy'])

            clr = CyclicLR(base_lr=initial_learning_rate, max_lr=max_learning_rate, step_size=step_size, mode='exp_range')
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[clr], verbose=0)
        
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        
        # Predictions
        y_pred = model.predict(X_test, verbose=0)
        y_pred_classes = (y_pred > 0.5).astype(int)

        # Compute confusion matrix
        conf_matrix = confusion_matrix(ytest, y_pred_classes)
        
        # Clear session and delete model to free up memory
        tf.keras.backend.clear_session()
        del model
        gc.collect()
        
        return accuracy, conf_matrix
        
    elif theta["model"] == 2:
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(np.shape(X_train)[1], np.shape(X_train)[2])),
            tf.keras.layers.Dense(theta["neuron1"], activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
    # Define your model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(tf.keras.layers.Dense(theta["neuron1"], activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    
    if theta["layers"] == 6:
        model.add(tf.keras.layers.Dense(theta["neuron2"], activation='relu'))
        model.add(tf.keras.layers.Dropout(0.2))
    
    elif theta["layers"] == 8:
        model.add(tf.keras.layers.Dense(theta["neuron2"], activation='relu'))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(theta["neuron1"], activation='relu'))
        model.add(tf.keras.layers.Dropout(0.2))
    
    
    # Add the output layer for binary classification
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    epochs = 300
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
        decay_steps = tf.constant(50, dtype=tf.int64)
        decay_rate = 0.9
        
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=fNirs_LRSchedule(
                initial_learning_rate=initial_learning_rate,
                decay_steps=decay_steps,
                decay_rate=decay_rate,
            )
        )
        model.compile(optimizer=optimizer,
                      loss=loss_fn, 
                      metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
                
        
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[tensorboard_callback], verbose=0)
        
    elif theta["learning_rate"] == "clr":
        initial_learning_rate = 0.0001
        max_learning_rate = 0.001
        step_size = 20
        
        optimizer = tf.keras.optimizers.Adam()
    
        model.compile(optimizer=optimizer,
                      loss=loss_fn,
                      metrics=['accuracy'])

        clr = CyclicLR(base_lr=initial_learning_rate, max_lr=max_learning_rate, step_size=step_size, mode='exp_range')
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[clr], verbose=0)

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    y_pred_probs = model.predict(X_test)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()
    y_true = ytest
    conf_matrix = confusion_matrix(y_true, y_pred)
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[tensorboard_callback, clr], verbose=0)
    
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    tf.keras.backend.clear_session()
    del model
    gc.collect()
    
    return accuracy

    return accuracy, conf_matrix