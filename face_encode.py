# manually train the pictures and save face_encodings.pkl using pickle
import face_recognition
import pickle
import os

known_face_encodings = []
known_names = []

# Load images and generate encodings
def load_and_encode_faces(image_folder):
    for person_name in os.listdir(image_folder):
        person_folder = os.path.join(image_folder, person_name)
        if os.path.isdir(person_folder):
            for image_name in os.listdir(person_folder):
                image_path = os.path.join(person_folder, image_name)
                image = face_recognition.load_image_file(image_path)
                face_encoding = face_recognition.face_encodings(image)[0]
                known_face_encodings.append(face_encoding)
                known_names.append(person_name)

# Load images from the folder
load_and_encode_faces("face_data")

# Save the encodings to a file
face_data = {"encodings": known_face_encodings, "names": known_names}
with open("face_encodings.pkl", "wb") as f:
    pickle.dump(face_data, f)
    print("succdesss")