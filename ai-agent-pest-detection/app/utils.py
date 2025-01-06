# app/utils.py

import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_data_generators(train_dir, val_dir, image_size=(224, 224), batch_size=32):
    """
    Create training and validation data generators from directories.
    """
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    return train_generator, val_generator

def load_and_preprocess_image(file, image_size=(224, 224)):
    """
    Load an image file (from Flask file object or path) and preprocess it.
    """
    # If it's a Flask file, read its raw stream
    if hasattr(file, 'read'):
        file_bytes = file.read()
        img = tf.io.decode_image(file_bytes, channels=3)
    else:
        # If it's a file path
        img = tf.io.decode_image(tf.io.read_file(file), channels=3)

    # Resize to target size
    img = tf.image.resize(img, image_size)
    # Normalize
    img = img / 255.0

    return img.numpy()

def plot_training_curves(history):
    """
    Plot training and validation accuracy/loss curves.
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

def get_class_indices(directory_path):
    """
    Returns a dictionary mapping class names to indices as discovered
    by flow_from_directory's default alphabetical sorting.
    e.g. {"Healthy": 0, "Powdery": 1, "Rust": 2}
    """
    class_names = sorted([
        d for d in os.listdir(directory_path)
        if os.path.isdir(os.path.join(directory_path, d))
    ])
    class_indices = {class_name: idx for idx, class_name in enumerate(class_names)}
    return class_indices
