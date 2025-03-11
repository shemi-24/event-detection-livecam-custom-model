import os
import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
import pickle

# Initialize models
detector = MTCNN()  # Face detector
embedder = FaceNet()  # Face embedding model

dataset_path = "C:\kaggle_incident_detection\kaggle\images_dataset"  # Your dataset folder
embeddings = []
labels = []

# Loop through each person's folder
for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)

    if os.path.isdir(person_folder):  # Ensure it's a folder
        for img_name in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_name)
            img = cv2.imread(img_path)

            if img is None:
                continue  # Skip unreadable images
            
            # Convert to RGB for MTCNN
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = detector.detect_faces(img_rgb)

            if faces:
                x, y, w, h = faces[0]['box']  # Get bounding box of the first detected face
                face = img_rgb[y:y+h, x:x+w]  # Crop face
                face = cv2.resize(face, (160, 160))  # Resize for FaceNet

                # Get face embedding
                embedding = embedder.embeddings([face])[0]
                
                embeddings.append(embedding)
                labels.append(person_name)  # Save the person's name as label

# Save embeddings and labels for later use
data = {"embeddings": np.array(embeddings), "labels": np.array(labels)}
with open("face_embeddings.pkl", "wb") as f:
    pickle.dump(data, f)

print("Embeddings saved successfully!")
