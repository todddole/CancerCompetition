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

print ("imports done.")

train_df = pd.read_csv('histopathologic-cancer-detection/train_labels.csv')

print(train_df.head())
print(train_df.info())

# convert to text labels for plotting purposes
label_mapping = {0: 'false', 1: 'true'}
train_df['label'] = train_df['label'].map(label_mapping)

# Add file extension to IDs to match file names
train_df['id'] = train_df['id'].apply(lambda x: x + '.tif')

batch_size = 128
target_size = (40, 40)
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    validation_split=0.25,
    vertical_flip = True,
    horizontal_flip = True,
    width_shift_range = 0.05,
    height_shift_range = 0.05
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
test_datagen = ImageDataGenerator(rescale=1.0/255)

# Use flow_from_directory to load test images
test_generator = test_datagen.flow_from_directory(
    directory="histopathologic-cancer-detection",
    target_size=target_size,
    batch_size=batch_size,
    class_mode=None,  # No labels for test data
    shuffle=False,    # Keep order for predictions
    classes=None      # Prevent it from expecting class subdirectories
)

# Validation generator (from the same train dataset with a different subset)
val_generator = train_datagen.flow_from_dataframe(
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
img_shape = (40, 40, 3)
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

model.save("model1.keras")
with open('training_log-model1.json', 'w') as f:
    json.dump(history.history, f)

with open('fine_tuning_log-model1.json', 'w') as f:
    json.dump(fine_tune_history.history, f)

