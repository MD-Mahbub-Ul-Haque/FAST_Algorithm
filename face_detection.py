import cv2
import face_recognition
import numpy as np
from pathlib import Path

# Load the pre-trained Haar Cascade face classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Path to the directory containing the face images for training
data_path = Path("training_data")

# Function to get the training data and labels
def get_training_data():
    faces = []
    labels = []
    
    for label, person in enumerate(data_path.glob('*')):
        for img_path in person.glob('*.jpg'):
            img = face_recognition.load_image_file(img_path)
            encoding = face_recognition.face_encodings(img)[0]
            faces.append(encoding)
            labels.append(label)

    return faces, labels

# Train the recognizer with the training data
faces, labels = get_training_data()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Recognize the face
        face_roi = gray[y:y + h, x:x + w]
        encoding = face_recognition.face_encodings(face_roi)
        
        if encoding:
            match = face_recognition.compare_faces(faces, encoding[0])
            
            if any(match):
                label = labels[match.index(True)]
                name = str(data_path.glob('*')[label]).split("\\")[-1]
                cv2.putText(frame, f'{name}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'Unknown', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Face Recognition', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
