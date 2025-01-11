import tensorflow as tf
from tensorflow.keras import layers, models

def create_cnn_model(input_shape, num_classes):
    """
    Creates a simple CNN model.

    Parameters:
    - input_shape: Shape of the input images (e.g., (224, 224, 3) for RGB images)
    - num_classes: Number of output classes for classification

    Returns:
    - model: Compiled Keras model
    """

    model = models.Sequential()

    # Add Convolutional Layer with MaxPooling
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Flatten the output of the convolution layers
    model.add(layers.Flatten())
    
    # Add Fully Connected Layers (Dense Layers)
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))  # Output layer with 'softmax' for classification
    
    return model
