# ImaginAIry - Image Classification Model

## Overview
ImaginAIry is a custom-built Convolutional Neural Network (CNN) model designed for image classification tasks. This project aims to train a deep learning model from scratch on your own dataset for classifying images into different categories. The model uses CNNs, a popular architecture for image-related tasks.

## Features
- **Image Classification**: Classify images into predefined categories.
- **Train from Scratch**: No pre-trained models—train your own model.
- **Customizable Architecture**: Modify the number of layers, neurons, and other parameters.
- **Training and Testing Pipelines**: Set up a training pipeline to train the model and a separate pipeline to evaluate it on unseen data.

## File Structure

- **`data/`**: This directory contains the image dataset. Organize it into subdirectories like `train/` and `test/`.
- **`src/`**: Source code files for the model. `cnn_model.py` contains the model architecture, and `utils.py` contains data loading and preprocessing functions.
- **`outputs/`**: Folder to store model outputs like training logs, checkpoints, and predictions.
- **`requirements.txt`**: Contains all the Python dependencies required for the project.
- **`main.py`**: Script that runs the training and evaluation of the model.

## TODO List

- [x] Set up the project structure.
- [x] Build the CNN model architecture from scratch (without using pre-trained models).
- [ ] Implement data preprocessing functions (loading and augmentation).
- [ ] Train the model on the image dataset.
- [ ] Evaluate the model using testing data and visualize performance.
- [ ] Implement model saving and loading functions.
- [ ] Optimize model performance (tune hyperparameters).
- [ ] Create a custom dataset loader for different image formats.

## Future Enhancements
- Experiment with more advanced CNN architectures (e.g., ResNet).
- Implement data augmentation for better generalization.
- Add support for multi-class and multi-label classification.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
