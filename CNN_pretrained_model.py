import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from CNN_data_prep import *

# Load MobileNetV2 pre-trained on ImageNet without the top layer
base_model = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet')

# Freeze the convolutional base to prevent weights from being updated during training
base_model.trainable = False

# Create the classification head
global_average_layer = GlobalAveragePooling2D()
prediction_layer = Dense(1, activation='sigmoid')

# Build the model
inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = base_model(inputs, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = Model(inputs, outputs)

# Summary of the model
model.summary()


initial_learning_rate = 0.001
lr_schedule = ExponentialDecay(
    initial_learning_rate,
    decay_steps=100,
    decay_rate=0.96,
    staircase=True)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss='binary_crossentropy',
    metrics=['accuracy'])


model.fit(train_generator, epochs=10, validation_data=validation_generator)

val_loss, val_accuracy = model.evaluate(validation_generator)
print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy*100:.2f}%")


# Fine-Tuning
# Unfreeze the base model
base_model.trainable = True

# Fine-tune from this layer onwards
fine_tune_at = 50

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),  # Lower learning rate for fine-tuning
    loss='binary_crossentropy',
    metrics=['accuracy'])

# Fine-tuning should be done with a smaller number of epochs
fine_tune_epochs = 5
total_epochs = 10 + fine_tune_epochs  # Total = initial epochs + fine-tuning epochs

model.fit(train_generator,
          epochs=total_epochs,
          initial_epoch=10,  # Continue from previous training
          validation_data=validation_generator)

# Evaluate model post fine-tuning
val_loss, val_accuracy = model.evaluate(validation_generator)
print(f"Post-Fine-Tuning Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy*100:.2f}%")