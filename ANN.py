import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC
import gc
import numpy as np
from clr_callback import CyclicLR  # Assuming you have a CyclicLR implementation
from seed import set_seeds
set_seeds()

def load_pretrained_weights(model, weights_path):
    try:
        pretrained_model = tf.keras.models.load_model(weights_path, compile=False)
        
        # Load weights from 'pre_dense_1' layer of the pretrained model to 'dense_1' layer of the classifier model
        if 'pre_dense_1' in pretrained_model.layers[-4].name:  # Check the layer name in the pretrained model
            model.get_layer('dense_1').set_weights(pretrained_model.get_layer('pre_dense_1').get_weights())

        # Load weights from 'pre_dense_2' layer of the pretrained model to 'dense_2' layer of the classifier model
        if 'pre_dense_2' in pretrained_model.layers[-3].name:  # Check the layer name in the pretrained model
            model.get_layer('dense_2').set_weights(pretrained_model.get_layer('pre_dense_2').get_weights())

        print("Pretrained weights loaded successfully for middle layers.")
    except Exception as e:
        print(f"Error loading pretrained weights: {e}")


def extract_features(X, model):
    dummy_data = tf.zeros((1, *X.shape[1:]))
    model.predict(dummy_data)
    feature_extractor = tf.keras.models.Model(inputs=model.input, outputs=model.layers[-2].output)
    features = feature_extractor.predict(X)
    return features

class fNirs_LRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, decay_steps, decay_rate):
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
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
    weights_path = "encoder_weights.h5"

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

    if theta["model"] == 1:
        # Define your model
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(X_train.shape[1], X_train.shape[2])),
            tf.keras.layers.Dense(200, activation='relu', name='dense_1'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(100, activation='relu', name='dense_2'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        # Load pretrained weights if the flag is set and path is provided
        if theta.get("use_transfer_learning", False) and weights_path:
            load_pretrained_weights(model, weights_path)

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
            model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

        # Ensure the model is built
        dummy_data = tf.zeros((1, X_train.shape[1], X_train.shape[2]))
        model(dummy_data)  # This will build the model

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
        conf_matrix = confusion_matrix(ytest, y_pred)

        # Clear session and delete model to free up memory
        tf.keras.backend.clear_session()
        del model
        gc.collect()

        return accuracy, conf_matrix
