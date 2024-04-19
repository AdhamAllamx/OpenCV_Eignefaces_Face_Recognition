import numpy as np 
import matplotlib.pyplot as plt  
import os 
import cv2 

training_data_folder_path = 'dataset/training-data'
test_data_folder_path = 'dataset/test-data'

# Define human-readable names for your classes
class_names = {
    'Junichiro_Koizumi': 0,
    'Alvaro_Uribe': 1,
    'George_Robertson': 2,
    'George_W_Bush': 3,
    'Atal_Bihari_Vajpayee': 4,
    'Amelia_Vega': 5,
    'Ana_Guevara': 6,
    # Add more names as needed
}

haarcascade_frontalface = 'opencv_xml_files/haarcascade_frontalface.xml'

def detect_face(input_img):
    image = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('opencv_xml_files/haarcascade_frontalface.xml')
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.2, minNeighbors=5)
    if len(faces) == 0:
        return None, None
    (x, y, w, h) = faces[0]
    return image[x:x+w, y:y+h], faces[0]

def training_data_set(training_data_folder_path):
    face_labels = [] 
    detected_faces = []
    training_image_directory = os.listdir(training_data_folder_path)

    for directory_name in training_image_directory:
        image_label = class_names[directory_name]  # Get the numerical label from class_names dictionary
        training_image_path = os.path.join(training_data_folder_path, directory_name)
        training_images_names = os.listdir(training_image_path)

        for image_name in training_images_names:
            image_path = os.path.join(training_image_path, image_name)
            image = cv2.imread(image_path)
            face, rectangle = detect_face(image)
            if face is not None:
                resized_face = cv2.resize(face, (255, 255), interpolation=cv2.INTER_AREA)
                detected_faces.append(resized_face)
                face_labels.append(image_label)
    return detected_faces, face_labels

detected_faces, face_labels = training_data_set(training_data_folder_path)

print("Total faces:", len(detected_faces))
print("Total labels:", len(face_labels))

# Print the labels to verify correctness
print("Labels:", face_labels)

eigenfaces_recognizer = cv2.face.EigenFaceRecognizer_create()  # Adjust num_components and threshold

eigenfaces_recognizer.train(detected_faces, np.array(face_labels))

def draw_rectangle(test_image, rectangle_coordinates):
    (x, y, width, height) = rectangle_coordinates 
    cv2.rectangle(test_image, (x, y), (x+width, y+height), (0, 255, 0), 2)

def draw_text_names(test_image, label_text, x, y):
    cv2.putText(test_image, label_text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

def prediction_test(test_image):
    detected_face, rectangle_coordinates = detect_face(test_image)
    resized_test_image = cv2.resize(detected_face, (128, 128), interpolation=cv2.INTER_AREA)
    label = eigenfaces_recognizer.predict(resized_test_image)

    # Print the predicted label to debug
    print("Predicted Label:", label)

    label_text = list(class_names.keys())[list(class_names.values()).index(label[0])]  # Get the corresponding class name
    draw_rectangle(test_image, rectangle_coordinates)
    draw_text_names(test_image, label_text, rectangle_coordinates[0], rectangle_coordinates[1]-5)
    return test_image, label_text

test_image = cv2.imread("dataset/test-data/Alvaro_Uribe/George_Robertson_0022.jpg")  # Adjust path accordingly

predicted_image, label = prediction_test(test_image)

figure = plt.figure()
ax1 = figure.add_axes((0.1, 0.2, 0.8, 0.7))
ax1.set_title('Actual class: ' + label + ' | Predicted class: ' + label)
plt.axis("off")
plt.imshow(cv2.cvtColor(predicted_image, cv2.COLOR_BGR2RGB))
plt.show()
