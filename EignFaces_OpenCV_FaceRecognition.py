import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

# Define paths and class names
training_data_folder_path = 'dataset/training-data'
class_names = {
    'Junichiro_Koizumi': 0,
    'Alvaro_Uribe': 1,
    'George_Robertson': 2,
    'George_W_Bush': 3,
    'Atal_Bihari_Vajpayee': 4,
    'Amelia_Vega': 5,
    'Ana_Guevara': 6,
}
haarcascade_frontalface = 'opencv_xml_files/haarcascade_frontalface.xml'

# Function to detect faces in images
def detect_face(input_img):
    image = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(haarcascade_frontalface)
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.2, minNeighbors=5)
    if len(faces) == 0:
        return None, None
    (x, y, w, h) = faces[0]
    return image[x:x+w, y:y+h], faces[0]

# Function to load and preprocess training data
def load_training_data(training_data_folder_path):
    face_labels = []
    detected_faces = []

    for directory_name in os.listdir(training_data_folder_path):
        image_label = class_names[directory_name]
        training_image_path = os.path.join(training_data_folder_path, directory_name)
        training_images_names = os.listdir(training_image_path)

        for image_name in training_images_names:
            image_path = os.path.join(training_image_path, image_name)
            image = cv2.imread(image_path)
            face, rectangle = detect_face(image)
            if face is not None:
                resized_face = cv2.resize(face, (121, 121), interpolation=cv2.INTER_AREA)
                normalized_face = resized_face / 255.0
                detected_faces.append(normalized_face)
                face_labels.append(image_label)
    
    return detected_faces, face_labels

# Load training data
detected_faces, face_labels = load_training_data(training_data_folder_path)
print("Total faces:", len(detected_faces))
print("Total labels:", len(face_labels))
print("Labels:", face_labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(detected_faces, face_labels, test_size=0.2, random_state=42)

# Define the pipeline with PCA and KNN
pipeline = Pipeline([
    ('pca', PCA()),  # PCA
    ('knn', KNeighborsClassifier())  # KNN
])

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'pca__n_components': [min(len(X_train), len(X_train[0])), 50, 100],  # Experiment with different values
    'knn__n_neighbors': [3, 5, 7],  # Experiment with different values
    'knn__metric': ['euclidean', 'manhattan']  # Experiment with different distance metrics
}

# Perform grid search for hyperparameter tuning
grid_search = GridSearchCV(pipeline, param_grid, cv=5, verbose=1)
grid_search.fit(X_train, y_train)

# Print the best hyperparameters found
print("Best hyperparameters:", grid_search.best_params_)

# Retrieve the best estimator
best_estimator = grid_search.best_estimator_

# Evaluate the model
train_accuracy = best_estimator.score(X_train, y_train)
test_accuracy = best_estimator.score(X_test, y_test)
print("Train Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)

# Function to predict labels using the best estimator
def predict_eigenfaces(test_image, classifier):
    detected_face, _ = detect_face(test_image)
    resized_test_image = cv2.resize(detected_face, (121, 121), interpolation=cv2.INTER_AREA)
    projected_test_data = best_estimator.named_steps['pca'].transform(resized_test_image.flatten().reshape(1, -1))
    label = classifier.predict(projected_test_data)[0]
    label_text = list(class_names.keys())[list(class_names.values()).index(label)]
    return label_text

# Example usage of the prediction function
test_image = cv2.imread("dataset/test-data/Ana_Guevara/Ana_Guevara_0006.jpg")
label = predict_eigenfaces(test_image, best_estimator)
print("Predicted Label (Eigenfaces with PCA):", label)

# Function to draw rectangle and text on images
def draw_rectangle(test_image, rectangle_coordinates):
    (x, y, width, height) = rectangle_coordinates 
    cv2.rectangle(test_image, (x, y), (x+width, y+height), (0, 255, 0), 2)

def draw_text_names(test_image, label_text, x, y):
    cv2.putText(test_image, label_text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

# Display the image with predicted label
draw_rectangle(test_image, (0, 0, test_image.shape[1], test_image.shape[0]))
draw_text_names(test_image, label, 0, 20)
plt.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
