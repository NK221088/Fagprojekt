import tensorflow as tf
from load_data_function import load_data
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import TensorBoard
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import warnings
from clr_callback import CyclicLR
import gc
import tensorflow as tf

# Suppress all warnings
warnings.filterwarnings("ignore")

# Your existing code here


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
    model.predict(X)
    feature_extractor = tf.keras.models.Model(inputs=model.input, outputs=model.layers[-2].output)
    features = feature_extractor.predict(X)
    return features

def load_pretrained_weights(model, weights_path):
    try:
        # Load the model without loading its architecture
        pretrained_model = tf.keras.models.load_model(weights_path, compile=False)

        # Extract weights from the pretrained model
        pretrained_weights = pretrained_model.layers[3].get_weights()

        # Set the weights of the specific layer in the main model
        model.layers[3].set_weights(pretrained_weights)

        print("Pretrained weights loaded successfully.")
    except Exception as e:
        print(f"Error loading pretrained weights: {e}")




def ANN_classifier(Xtrain, ytrain, Xtest, ytest, theta):
    
    weights_path = "encoder_weights.h5"
    
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
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(np.shape(X_train)[1], np.shape(X_train)[2])))
    model.add(tf.keras.layers.Dense(60, activation='relu'))  # Match the pretrained model's architecture
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(150, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
        
    if theta["layers"] == 8:
        model.add(tf.keras.layers.Dense(theta["neurons2"], activation='relu'))
        model.add(tf.keras.layers.Dropout(0.2))
        

    # Add the output layer for binary classification
    model.add(tf.keras.layers.Dense(1, activation='sigmoid')) 

    # Load pretrained weights if provided
    if weights_path:
        # Build the model first by calling it on some dummy data
        dummy_data = tf.zeros((1, np.shape(X_train)[1], np.shape(X_train)[2]))
        model(dummy_data)  # This will build the model
        load_pretrained_weights(model, weights_path)
        
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False) # We use BinaryCrossentropy as there is only two classes
    epochs = 150
    batch_size = 100

    log_dir = "logs/"
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    if theta["learning_rate"] == "decrease":
        initial_learning_rate = 0.01
        decay_steps = 10 #tf.constant(10, dtype=tf.int64)
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
                
    elif theta["learning_rate"] == "clr":
        initial_learning_rate = 0.001
        max_learning_rate = 0.01
        step_size = 10
        
        optimizer = tf.keras.optimizers.Adam()
    
        model.compile(optimizer=optimizer,
                    loss=loss_fn,
                    metrics=['accuracy'])

        clr = CyclicLR(base_lr=initial_learning_rate, max_lr=max_learning_rate, step_size=step_size, mode='exp_range')
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[tensorboard_callback, clr], verbose=0)

    # Run a dummy forward pass to build the model
    model.predict(X_train)
    
    # Extract features from the trained network
    train_features = extract_features(X_train, model)
    test_features = extract_features(X_test, model)

    # Train an SVM on the extracted features
    svm = SVC(kernel='linear')
    svm.fit(train_features, ytrain)

    # Evaluate the SVM
    y_pred = svm.predict(test_features)
    accuracy = accuracy_score(ytest, y_pred)

    # Clear session and delete model to free up memory
    tf.keras.backend.clear_session()
    del model
    gc.collect()

    return accuracy

"""
# Plot the training loss
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper right')
plt.show()
"""