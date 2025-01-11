import tensorflow as tf
from src.cnn_model import create_cnn_model
from src.utils import load_data, preprocess_data, plot_training_history
from tensorflow.keras.optimizers import Adam
import os

# Set up directories for saving model and logs
checkpoint_dir = "outputs/checkpoints"
log_dir = "outputs/logs"
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# Load and preprocess data
train_data, val_data, test_data = load_data("data/train", "data/validation", "data/test")
train_data, val_data, test_data = preprocess_data(train_data, val_data, test_data)

# Define the model
model = create_cnn_model(input_shape=(224, 224, 3), num_classes=10)  # Adjust num_classes as per dataset

# Compile the model
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,  # Number of epochs to train
    callbacks=[tf.keras.callbacks.ModelCheckpoint(checkpoint_dir + '/model_{epoch}.h5', save_best_only=True),
               tf.keras.callbacks.TensorBoard(log_dir)]
)

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(test_data)
print(f"Test Accuracy: {test_acc}")

# Visualize the training history (Loss and Accuracy)
plot_training_history(history)
