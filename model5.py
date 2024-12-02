import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization, GlobalAveragePooling2D, Input
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator


import os
import json

def center_crop(image, crop_size=(32, 32)):
    """
    Crop the center of the image to the specified size.
    """
    crop_height, crop_width = crop_size
    height, width, _ = image.shape

    start_x = (width - crop_width) // 2
    start_y = (height - crop_height) // 2

    cropped_image = image[start_y:start_y + crop_height, start_x:start_x + crop_width, :]
    return cropped_image


train_df = pd.read_csv('histopathologic-cancer-detection/train_labels.csv')

print(train_df.head())
print(train_df.info())

# convert to text labels for plotting purposes
label_mapping = {0: 'false', 1: 'true'}
train_df['label'] = train_df['label'].map(label_mapping)

# Add file extension to IDs to match file names
train_df['id'] = train_df['id'].apply(lambda x: x + '.tif')

batch_size = 128
target_size = (32, 32)
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    validation_split=0.25,
    vertical_flip = True,
    horizontal_flip = True,
    width_shift_range = 0.05,
    height_shift_range = 0.05,
    preprocessing_function=lambda img: center_crop(img, crop_size=(32, 32))
)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    directory="histopathologic-cancer-detection/train",
    x_col = "id",
    y_col = "label",
    target_size = target_size,
    batch_size = batch_size,
    class_mode = "binary",
    subset = 'training',
    shuffle = True
)

# Test data generator (only rescaling, no augmentation needed)
test_datagen = ImageDataGenerator(
        rescale=1.0/255,
        preprocessing_function=lambda img: center_crop(img, crop_size=(32, 32))
        )

# Use flow_from_directory to load test images
test_generator = test_datagen.flow_from_directory(
    directory="histopathologic-cancer-detection",
    target_size=target_size,
    batch_size=batch_size,
    class_mode=None,  # No labels for test data
    shuffle=False,    # Keep order for predictions
    classes=None      # Prevent it from expecting class subdirectories
)

val_datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.25,
    preprocessing_function=lambda img: center_crop(img, crop_size=(32, 32))
)

# Validation generator (from the same train dataset with a different subset)
val_generator = val_datagen.flow_from_dataframe(
    train_df,
    directory="histopathologic-cancer-detection/train",
    x_col="id",
    y_col="label",
    target_size=target_size,
    batch_size=batch_size,
    class_mode="binary",
    subset="validation",
    shuffle=False
)

# Define the image shape
img_shape = (32, 32, 3)
# Load the DenseNet121 model pre-trained on ImageNet, excluding the top layers
base_model = DenseNet121(include_top=False, weights='imagenet', input_shape=img_shape)

# Freeze the base model layers to prevent them from being updated during initial training
base_model.trainable = False

# Add custom layers on top of the base model
inputs = Input(shape=img_shape)
x = preprocess_input(inputs)  # Preprocess input for DenseNet
x = base_model(x, training=False)  # Pass through base model
x = GlobalAveragePooling2D()(x)  # Pooling layer to reduce dimensionality
x = Dense(256, activation='relu')(x)  # Fully connected layer
x = Dense(128, activation='relu')(x)  # Another fully connected layer
outputs = Dense(1, activation='sigmoid')(x)  # Output layer for binary classification

# Create the complete model
model = Model(inputs, outputs)

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Summary of the model
model.summary()

# Train the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,  # Adjust as needed
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=val_generator.samples // val_generator.batch_size
)


# Fine-tuning: Unfreeze the base model and retrain
base_model.trainable = True
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),  # Lower learning rate for fine-tuning
              loss='binary_crossentropy',
              metrics=['accuracy'])

fine_tune_history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,  # Adjust as needed
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=val_generator.samples // val_generator.batch_size
)

model.save("model5.keras")

with open('training_log-model5.json', 'w') as f:
    json.dump(history.history, f)

with open('fine_funing_log-model5.json', 'w') as f:
    json.dump(fine_tune_history.history, f)