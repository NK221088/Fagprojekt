import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC
import gc
import numpy as np
from clr_callback import CyclicLR  # Assuming you have a CyclicLR implementation
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
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
    
    #Performing standardization
    epsilon = 1e-8
    std = np.std(Xtrain, axis=0)
    std[std == 0] = epsilon     # Replace zero standard deviations with epsilon
    Xtest = (Xtest - np.mean(Xtrain, axis=0)) / std
    Xtrain = (Xtrain - np.mean(Xtrain, axis=0)) / std

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

            model.compile(optimizer=optimizer,
                        loss=loss_fn,
                        metrics=['accuracy'])
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

        elif theta["learning_rate"] == "clr":
            initial_learning_rate = 0.001
            max_learning_rate = 0.0025
            step_size = 50

            optimizer = tf.keras.optimizers.Adam()

            model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

            clr = CyclicLR(base_lr=initial_learning_rate, max_lr=max_learning_rate, step_size=step_size, mode='exp_range')
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[clr], verbose=0)

        if theta["use_svm"] == True:
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
            return accuracy, conf_matrix, (y_pred, ytest)
        
        else:
            loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
            y_pred_probs = model.predict(X_test)
            y_pred = (y_pred_probs > 0.5).astype(int).flatten()
            y_true = ytest
            conf_matrix = confusion_matrix(y_true, y_pred)

            tf.keras.backend.clear_session()
            del model
            gc.collect()
            return accuracy, conf_matrix, (y_pred, ytest)

    elif theta["model"] == 2:
            model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(np.shape(X_train)[1], np.shape(X_train)[2])),
            BatchNormalization(),
            Dropout(0.5),
            Conv1D(filters=128, kernel_size=3, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            LSTM(theta["neuron2"], return_sequences=True, kernel_regularizer=l2(0.001)),
            LSTM(theta["neuron1"], kernel_regularizer=l2(0.001)),
            Dense(1, activation='sigmoid')
        ])
            
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
                initial_learning_rate = 0.001
                max_learning_rate = 0.0025
                step_size = 50

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

            tf.keras.backend.clear_session()
            del model
            gc.collect()
            return accuracy, conf_matrix, (y_pred, ytest)
    
    elif theta["model"] == 3:
        model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(np.shape(X_train)[1], np.shape(X_train)[2])),
        tf.keras.layers.Dense(theta["neuron1"], activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer for binary classification. The units is 1, as the output of the sigmoid function represents the probability of belonging to the positive class
        ])
        
        loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        epochs = 100
        batch_size = 100
        initial_learning_rate = 0.009
        decay_steps = 20
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
        
        model.fit(X_train, y_train, epochs = epochs, batch_size = batch_size,verbose = 0)
        
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        y_pred_probs = model.predict(X_test)
        y_pred = (y_pred_probs > 0.5).astype(int).flatten()
        y_true = ytest
        conf_matrix = confusion_matrix(y_true, y_pred)

        tf.keras.backend.clear_session()
        del model
        gc.collect()
        return accuracy, conf_matrix, (y_pred, ytest)