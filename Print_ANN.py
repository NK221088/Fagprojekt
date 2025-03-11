import tensorflow as tf
from tensorflow.keras.utils import plot_model

# Define the input layer with a fixed batch size
inputs = tf.keras.Input(shape=(X_train.shape[1], X_train.shape[2]), batch_size=batch_size, name='InputLayer')

# Define the model architecture
x = tf.keras.layers.Flatten()(inputs)
x = tf.keras.layers.Dense(300, activation='relu', name='DenseLayer1')(x)
x = tf.keras.layers.Dropout(0.2, name='Dropout1')(x)
x = tf.keras.layers.Dense(200, activation='relu', name='DenseLayer2')(x)
x = tf.keras.layers.Dropout(0.2, name='Dropout2')(x)
x = tf.keras.layers.Dense(300, activation='relu', name='DenseLayer3')(x)
x = tf.keras.layers.Dropout(0.2, name='Dropout3')(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid', name='OutputLayer')(x)

# Create the model
model = tf.keras.Model(inputs=inputs, outputs=outputs, name='MyModel')

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

# Save the model using tf.saved_model.save
tf.saved_model.save(model, 'my_model')

# Visualize the model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True, expand_nested=True, dpi=200)
