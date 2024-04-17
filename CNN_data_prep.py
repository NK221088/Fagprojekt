from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the image size expected by MobileNet V2
IMG_SIZE = (224, 224)  # Resize target for images

# Set up the ImageDataGenerator for training
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values to [0, 1]
    #rotation_range=40,  # Randomly rotate images in the range (degrees, 0 to 180)
    #width_shift_range=0.2,  # Randomly translate images horizontally (fraction of total width)
    #height_shift_range=0.2,  # Randomly translate images vertically (fraction of total height)
    #shear_range=0.2,  # Randomly apply shearing transformations
    #zoom_range=0.2,  # Randomly zoom inside pictures
    #horizontal_flip=True,  # Randomly flip images horizontally
    #fill_mode='nearest'  # Strategy to fill newly created pixels, which can appear after a rotation or a width/height shift
)

# Set the path to the training directory
train_dir = 'fNIRS_images/train/'

train_generator = train_datagen.flow_from_directory(
    train_dir,  # Directory containing training data
    target_size=IMG_SIZE,
    batch_size=32,
    class_mode='binary'
)

# Set up the ImageDataGenerator for validation (no data augmentation here)
validation_datagen = ImageDataGenerator(rescale=1./255)

# Set the path to the validation directory
val_dir = 'fNIRS_images/val/'

validation_generator = validation_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=32,
    class_mode='binary'
)