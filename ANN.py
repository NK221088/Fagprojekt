import tensorflow as tf
from load_data_function import load_data
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import TensorBoard
from sklearn.decomposition import PCA

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

def ANN_classifier(Xtrain, ytrain, Xtest, ytest):
    
    # Allow memory growth for the GPU
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            # Set memory growth to avoid allocating all GPU memory at once
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except RuntimeError as e:
            print(e)
    

    
    X_train = tf.convert_to_tensor(Xtrain)
    y_train = tf.convert_to_tensor(ytrain)
    X_test = tf.convert_to_tensor(Xtest)
    y_test = tf.convert_to_tensor(ytest)
    
    model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(np.shape(X_train)[1], 3)),
    tf.keras.layers.Dense(200, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1000, activation='gelu'),
    tf.keras.layers.Dense(10, activation='tanh'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer for binary classification. The units is 1, as the output of the sigmoid function represents the probability of belonging to the positive class
    ])
    
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False) # We use BinaryCrossentropy as there is only two classes
    
    initial_learning_rate = 0.001
    decay_steps = tf.constant(10, dtype=tf.int64)
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
    
    plot_model(model, to_file='ANN_model_structure.png', show_shapes=True)

    log_dir = "logs/"
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Reshape input data to have only two dimensions
    X_train_flat = np.reshape(X_train, (np.shape(X_train)[0], -1))

    # Initialize PCA with desired number of components
    n_components = 3
    pca = PCA(n_components=n_components)

    # Fit PCA to the flattened training data
    pca.fit(X_train_flat)

    # Transform both training and testing data using the trained PCA
    X_train_pca = pca.transform(X_train_flat)
    X_test_pca = pca.transform(np.reshape(X_test, (np.shape(X_test)[0], -1)))

    
    model.fit(X_train_pca, y_train, epochs=50, callbacks=[tensorboard_callback])
    
    accuracy = model.evaluate(X_test_pca,  y_test, verbose=2)
    
    return accuracy