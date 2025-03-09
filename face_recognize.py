import cv2
import os
import face_recognition
import numpy as np
import pickle

face_data_path="face_data"
if not os.path.exists(face_data_path):
    os.makedirs(face_data_path)

cap=cv2.VideoCapture(0)    
person_name=input("enter the name:")

count=0
known_face_encodings=[]
known_names=[]

while count<10:
    ret,frame=cap.read()
    if not ret:
        break

    rgb_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    face_locations=face_recognition.face_locations(rgb_frame)

    if face_locations: # ithil faceinte coordinated indakum
        count+=1
        image_path=os.path.join(face_data_path,f"{person_name}_{count}.jpg")
        cv2.imwrite(image_path,frame)
        print(f"saved image{image_path}")

        # Encode the face and store the encoding
        encoding=face_recognition.face_encodings(rgb_frame,face_locations)
        if encoding:
            known_face_encodings.append(encoding[0])
            known_names.append(person_name)


        cv2.imshow("Capturing Faces", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# Save encodings to a file

with open("face_encodings.pkl","wb") as f:
    pickle.dump({"encodings":known_face_encodings,"names":known_names},f)
