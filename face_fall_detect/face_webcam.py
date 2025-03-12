import numpy as np
import pickle
import cv2
from mtcnn import MTCNN
from keras_facenet import FaceNet

with open("face_embeddings.pkl","rb") as f:
    data=pickle.load(f)
embeddings=data['embeddings']
labels=data["labels"]
print(len(embeddings))
print(labels)       

detecter=MTCNN()
embedder=FaceNet()
# Define cosine similarity function
def cosine_similarity(embedding1, embedding2):
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detecter.detect_faces(img_rgb)

    for face in faces:
        x, y, w, h = face['box']
        face_img = img_rgb[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (160, 160))

        # Get new embedding
        new_embedding = embedder.embeddings([face_img])[0]

        # Compare with known embeddings
        similarities = [cosine_similarity(new_embedding, emb) for emb in embeddings]

        # Find best match
        best_match_index = np.argmax(similarities)
        best_match_label = labels[best_match_index]
        best_match_score = similarities[best_match_index]

        # Display result
        if best_match_score > 0.6:  # Adjust threshold as needed
            label_text = f"{best_match_label} ({best_match_score:.2f})"
        else:
            label_text = "Unknown"

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()