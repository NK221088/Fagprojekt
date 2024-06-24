import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import plot_model
from fpdf import FPDF
from PIL import Image

# Example shape for X_train and theta
X_train = np.random.rand(100, 28, 28)  # Assuming a dataset with shape (100, 28, 28)
theta = np.arange(1, 301)  # Assuming theta is an array with at least 300 elements

# Define the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(tf.keras.layers.Dense(theta[299], activation='relu'))  # Adjusted index for correct access
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(theta[199], activation='relu'))  # Adjusted index for correct access
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(theta[299], activation='relu'))  # Adjusted index for correct access
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# Generate a plot of the model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# Load the image
image_path = 'model_plot.png'
image = Image.open(image_path)

# Create a PDF document
pdf = FPDF()
pdf.add_page()
pdf.image(image_path, x=10, y=10, w=pdf.w - 20)

# Save the PDF
pdf_output_path = "model_structure.pdf"
pdf.output(pdf_output_path)

print(f"PDF saved as {pdf_output_path}")
