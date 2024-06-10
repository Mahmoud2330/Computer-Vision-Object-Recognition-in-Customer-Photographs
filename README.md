# Computer-Vision-Object-Recognition-in-Customer-Photographs
Develop an object recognition model that can detect 4 different classes: food, building, landscape and people. 

## Project Overview
This project develops an object recognition model capable of categorizing images into four classes: food, buildings, landscapes, and people. Designed for a web-based company, the model enhances user experience and advertising effectiveness on the platform.

## Team Members
- Mahmoud Ibrahim Elsayed

## Problem Statement
The goal is to create a model that accurately classifies customer photographs to improve user interaction and optimize advertising strategies on a web platform.

## Dataset
The dataset consists of labeled images divided into:
- **Training Data:** For training the model.
- **Validation Data:** Unseen labeled data for validating the model.
- **Testing Data:** Unseen unlabeled data for testing the model.

## Technologies Used
- Python
- TensorFlow
- Keras
- OpenCV
- Pandas
- NumPy
- Matplotlib

## Features
### Preprocessing
Transforms images for model readiness through grayscale conversion, blurring, and edge detection.

### Data Handling and Augmentation
Uses `ImageDataGenerator` for robust training, including data augmentation and preprocessing.

### Model Training
Configures and trains a TensorFlow model, customizing loss functions and optimizers for optimal performance.

### Results Visualization
Provides tools to visualize raw and preprocessed images, and to present model predictions.

## Getting Started
These instructions will get the project up and running on your local machine for development and testing purposes.

### Prerequisites
Ensure you have Python and the necessary libraries installed:
```bash
pip install tensorflow keras opencv-python pandas numpy matplotlib
