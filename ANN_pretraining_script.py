import tensorflow as tf
import numpy as np
from load_data_function import load_data
from seed import set_seeds

set_seeds()

def pretrain_model(X, save_path):
    input_shape = X.shape[1:]
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.Dense(200, activation='relu', name='pre_dense_1'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(100, activation='relu', name='pre_dense_2'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(np.prod(input_shape), activation='sigmoid'),
        tf.keras.layers.Reshape(input_shape)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, X, epochs=50, batch_size=128, verbose=1)
    
    model.save(save_path)
    print(f"Model saved to {save_path}")

# Data set:
data_set = "fNirs_motor_full_data"
individuals = True

# Data processing:
short_channel_correction = True
negative_correlation_enhancement = True
interpolate_bad_channels = False

if individuals:
    all_epochs, data_name, all_data, freq, data_types, all_individuals = load_data(
        data_set=data_set, 
        short_channel_correction=short_channel_correction, 
        negative_correlation_enhancement=negative_correlation_enhancement, 
        individuals=individuals, 
        interpolate_bad_channels=interpolate_bad_channels
    )

# Combining all relevant data
additional_data = np.append(all_data["Tapping"], all_data["Control"], axis=0)

# Normalizing the data
additional_data = (additional_data - np.mean(additional_data, axis=0)) / np.std(additional_data, axis=0)

# Pretrain autoencoder and save weights
save_path = "encoder_weights.h5"
pretrain_model(additional_data, save_path)
