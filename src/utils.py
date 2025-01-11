import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(train_dir, val_dir, test_dir, batch_size=32):
    """
    Loads the training, validation, and test data using the ImageDataGenerator.

    Parameters:
    - train_dir: Directory containing training images
    - val_dir: Directory containing validation images
    - test_dir: Directory containing test images
    - batch_size: The batch size for the data generators

    Returns:
    - train_data: Generator for the training data
    - val_data: Generator for the validation data
    - test_data: Generator for the test data
    """

    # Initialize ImageDataGenerator for data augmentation
    train_datagen = ImageDataGenerator(rescale=1./255, 
                                       rotation_range=20, 
                                       width_shift_range=0.2, 
                                       height_shift_range=0.2, 
                                       shear_range=0.2, 
                                       zoom_range=0.2, 
                                       horizontal_flip=True, 
                                       fill_mode='nearest')

    val_test_datagen = ImageDataGenerator(rescale=1./255)

    # Load data from directories
    train_data = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),  # Resize images
        batch_size=batch_size,
        class_mode='sparse'  # 'sparse' for integer labels
    )

    val_data = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='sparse'
    )

    test_data = val_test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='sparse'
    )

    return train_data, val_data, test_data

def preprocess_data(train_data, val_data, test_data):
    """
    Preprocesses the data. This function can be extended for additional preprocessing steps.

    Parameters:
    - train_data, val_data, test_data: ImageDataGenerator objects

    Returns:
    - Preprocessed train, validation, and test data generators
    """
    # For now, no additional preprocessing steps are defined
    return train_data, val_data, test_data

def plot_training_history(history):
    """
    Plots the training loss and accuracy over epochs.

    Parameters:
    - history: History object returned by model.fit()
    """

    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

