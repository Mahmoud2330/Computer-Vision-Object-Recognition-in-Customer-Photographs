# Mahmoud Ibrahim Elsayed 20025336

import os
import cv2
import random
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from pathlib import Path
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# This is a preprocessing method that turns the image into grayscale, then does edge detection then blurs any noise that is visible in the image of the edge detection to show the prominent lines
def preprocessing(image_location):
    image = cv2.imread(image_location)
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grayscale, (5, 5), 0)
    edge_detection = cv2.Canny(blur, 100, 200)
    return edge_detection


# Display random images from a folder
def present_images(folder, amount_images=50):
    for i in range(amount_images):
        file = random.choice(os.listdir(folder))
        axis = plt.subplot(5, 10, i + 1)
        image = mpimg.imread(os.path.join(folder, file))
        plt.imshow(image)
    plt.show()


# Display preprocessed images from a folder
def present_preprocessing(folder, amount_images=50):
    for i in range(amount_images):
        file = random.choice(os.listdir(folder))
        image = mpimg.imread(os.path.join(folder, file))
        axis = plt.subplot(5, 10, i + 1)
        plt.imshow(image, cmap='gray')
    plt.show()


# Display images from specified categories in a root path
def present_categories(root_path, categories):
    for category in categories:
        present_images(os.path.join(root_path, category))


# Load and randomize dataset, returning a DataFrame
def load_randomise_dataset(folder_path):
    image_path = list(folder_path.glob(r"**/*.jpg"))
    label = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], image_path))
    final_images = pd.DataFrame({"image_info": image_path, "label": label}).astype("str")
    final_images = final_images.sample(frac=1).reset_index(drop=True)
    return final_images[['image_info', 'label']]


# Generate data using ImageDataGenerator and return a data flow
def generate_data(table, batch_size, parameters):
    generator = tf.keras.preprocessing.image.ImageDataGenerator(**parameters)
    return generator.flow_from_dataframe(
        dataframe=table,
        x_col="image_info",
        y_col="label",
        batch_size=batch_size,
        class_mode="categorical",
        target_size=(224, 224),
        color_mode="rgb",
        shuffle=True
    )


# Train the model and return the training history
def train_model(model, train_preprocess, valid_preprocess, epochs=36):
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['accuracy']
    )

    previous_model = model.fit(train_preprocess, epochs=epochs, validation_data=valid_preprocess)
    return previous_model


# Present predictions using the trained model
def present_predictions(model, test_preprocess, test_data):
    prediction = model.predict(test_preprocess)
    prediction = np.argmax(prediction, axis=1)
    locate_label = dict((m, n) for n, m in test_preprocess.class_indices.items())
    final_prediction = pd.Series(prediction).map(locate_label).values
    y_test = list(test_data.label)

    plt.figure(figsize=(40, 40))
    plt.style.use("classic")
    image_amount = (5, 10)

    plt.subplots_adjust(left=0.02, bottom=0.1, right=0.98, top=0.9, wspace=0.2, hspace=0.4)

    for i in range(1, (image_amount[0] * image_amount[1]) + 1):
        plt.subplot(image_amount[0], image_amount[1], i)
        plt.axis("Off")

        color = "blue"
        if test_data.label.iloc[i] != final_prediction[i]:
            color = "red"
            ##
        plt.title(f"True: {test_data.label.iloc[i]}\nPrediction: {final_prediction[i]}", color=color, fontsize = 9)
        plt.imshow(plt.imread(test_data['image_info'].iloc[i]), aspect = 'auto')

    plt.show()


categories = ['food', 'building', 'landscape', 'people']
root_path = '/Users/mahmoudibrahim/Documents/20025336_CV_Project/CV_Project_Dataset/training'
present_categories(root_path, categories)

# Define image location and label names
image_location = '/Users/mahmoudibrahim/Documents/20025336_CV_Project/CV_Project_Dataset/training'
label_names = os.listdir(image_location)
print(f"Label names: {label_names}")
print(f"Number of labels: {len(label_names)}")

# Filter label names to include only directories
label_names = [label_name for label_name in label_names if os.path.isdir(os.path.join(root_path, label_name))]

# Display label information
print("Label names:", label_names)
print("Number of labels:", len(label_names))

# Calculate the amount of images for each category
amount_of_images = {label_name: len(os.listdir(os.path.join(root_path, label_name))) for label_name in label_names}
print("Amount of images:", amount_of_images)

# Load and randomize training dataset
folder_train_data = Path(r"/Users/mahmoudibrahim/Documents/20025336_CV_Project/CV_Project_Dataset/training")
train_data = load_randomise_dataset(folder_train_data)
print(train_data)

# Load and randomize validation dataset
folder_valid_data = Path(r"/Users/mahmoudibrahim/Documents/20025336_CV_Project/CV_Project_Dataset/validation")
valid_data = load_randomise_dataset(folder_valid_data)
print(valid_data)

# Load and randomize testing dataset
folder_test_data = Path(r"/Users/mahmoudibrahim/Documents/20025336_CV_Project/CV_Project_Dataset/testing")
test_data = load_randomise_dataset(folder_test_data)
print(test_data)

batch_size = 40
# Define data generator parameters
training_data_generator_parameters = {
    'rescale': 1. / 255, 'width_shift_range': 0.4, 'height_shift_range': 0.4, 'zoom_range': 0.3,
    'horizontal_flip': True, 'shear_range': 0.4, 'validation_split': 0.2, 'fill_mode': 'nearest'
}

validation_parameters = {'rescale': 1. / 255}
testing_parameters = {'rescale': 1. / 255}

# Generate data flows using ImageDataGenerator
training_data_generator = generate_data(train_data, batch_size, training_data_generator_parameters)
validation_data_generator = generate_data(valid_data, batch_size, validation_parameters)
testing_data_generator = generate_data(test_data, batch_size, testing_parameters)

# Get class indices
c_dictionary = training_data_generator.class_indices
c_list = list(c_dictionary.keys())
print(c_list)

# Define the neural network model
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[224, 224, 3]),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(70, activation='relu'),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dense(20, activation='relu'),
    keras.layers.Dense(4, activation='softmax')
])

# Display a summary of the neural network model architecture
model.summary()

# Train the model using the training and validation data generators
Train_the_model = train_model(model, training_data_generator, validation_data_generator, epochs=36)

# Extract accuracy values from the training history so that it is ploted on the graph for the loss and accuracy as the y axis with the epoch as the x axis
accuracies_epochs = Train_the_model.history['accuracy']
epochs = range(1, 37)
sns.lineplot(x=epochs, y = accuracies_epochs)

accuracies_epochs = Train_the_model.history['loss']
epochs = range(1, 37)
sns.lineplot(x=epochs, y = accuracies_epochs)

# Present predictions using the trained model on the testing dataset
present_predictions(model, testing_data_generator, test_data)

# Evaluate and print the testing loss and accuracy of the model
testing_loss, testing_accuracy = model.evaluate(testing_data_generator)
print(f'Testing Loss: {testing_loss}, Testing Accuracy: {testing_accuracy}')