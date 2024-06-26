import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Resizing
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
import gc

gpus = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpus))
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

def make_dataset(X, y, batch_size=32, augment=False):
    min_val = np.min(X)
    max_val = np.max(X)

    X_normalized = (X - min_val) / (max_val - min_val)

    X_normalized = np.expand_dims(X_normalized, axis=-1)
    X_normalized = np.repeat(X_normalized, 3, axis=-1)

    # Convert to TensorFlow Dataset object
    dataset = tf.data.Dataset.from_tensor_slices((X_normalized, y))

    if augment:
        dataset = dataset.map(lambda x, y: (tf.image.random_flip_left_right(x), y))
    dataset = dataset.map(lambda x, y: (tf.image.resize(x, [224, 224]), y))  # Resize images
    dataset = dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def plot_samples(X, n=5):
    plt.figure(figsize=(10, 2))
    for i in range(n):
        ax = plt.subplot(1, n, i + 1)
        plt.imshow(X[i].squeeze(), cmap='gray')
        plt.axis('off')
    plt.show()


def CNN_classifier(Xtrain, ytrain, Xtest, ytest, theta):
    train_dataset = make_dataset(Xtrain, ytrain, batch_size=theta["batch_size"], augment=True)
    val_dataset = make_dataset(Xtest, ytest, batch_size=theta["batch_size"])

    base_learning_rate = theta["base_learning_rate"]
    # lr_schedule = ExponentialDecay(initial_learning_rate, decay_steps=100, decay_rate=0.96, staircase=True)

    # Pretrained CNN model:f
    # Load MobileNetV2 pre-trained on ImageNet without the top layer
    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False

    # Building the model
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = Dense(1, activation='sigmoid', dtype=tf.float32)(x)
    model = Model(inputs, outputs)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Initial training
    model.fit(train_dataset, epochs=5, validation_data=val_dataset)

    # Unfreeze the base model for fine-tuning
    base_model.trainable = True

    # Fine-tune from this layer onwards
    fine_tune_at = theta["number_of_layers"]

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    # Re-compile the model for fine-tuning
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate / 10),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Fine-tuning
    model.fit(train_dataset, epochs=5, initial_epoch=5, validation_data=val_dataset)

    accuracy = model.evaluate(val_dataset)
    
    # Clear session and delete model to free up memory
    tf.keras.backend.clear_session()
    del model
    gc.collect()
    
    return accuracy
