import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Resizing
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import numpy as np
import matplotlib.pyplot as plt

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
    train_dataset = make_dataset(Xtrain, ytrain, batch_size=32, augment=True)
    val_dataset = make_dataset(Xtest, ytest, batch_size=32)

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
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, outputs)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Initial training
    model.fit(train_dataset, epochs=5, validation_data=val_dataset)

    # Unfreeze the base model for fine-tuning
    base_model.trainable = True

    # Fine-tune from this layer onwards
    fine_tune_at = 100

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
    return accuracy
