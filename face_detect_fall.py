# this code is face_recognize.py(this will iam train the face and store using pickle) and testtfile.py -integration
import cv2
import os
import face_recognition
import numpy as np
import pickle
import json
from ultralytics import YOLO
from cvzone.Utils import putTextRect
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import winsound

# Load face encodings from file
with open("face_encodings.pkl", "rb") as f:
    face_data = pickle.load(f)
    known_face_encodings = face_data["encodings"]
    known_names = face_data["names"]

# Email alert function
def send_email(to_email, subject, message):
    sender_email = "support@zoftcares.in"  
    sender_password = "U_Dr*0R($q&?"    

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(message, "plain"))

    try:
        server = smtplib.SMTP_SSL("mail.zoftcares.in", 25)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, to_email, msg.as_string())
        server.quit()
        print("Email sent successfully!")
        return True
    except Exception as e:
        print(f"Email sending failed: {e}")
        return False

# Load trained YOLO model
model = YOLO('C:\\kaggle_incident_detection\\kaggle\\runs\\detect\\train_dataset\\weights\\best.pt')

# Open video file
cap = cv2.VideoCapture('hashir2_fall.mp4')

previous_y_positions = {}
fall_threshold = 130  
email_sent = False  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  

    # Run YOLO inference
    results = model(frame)

    detected_name = "Unknown"  # Default if no face is recognized

    # Face recognition
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # Speed up face processing
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances) if matches else None

        if best_match_index is not None and matches[best_match_index]:
            detected_name = known_names[best_match_index]

    for result in results:
        boxes = result.boxes  

        for box in boxes:
            class_id = int(box.cls)  
            confidence = float(box.conf)  
            x1, y1, x2, y2 = map(int, box.xyxy[0])  

            class_name = model.names[class_id]
            print(f"Detected: {class_name} (Confidence: {confidence:.2f})")  

            if confidence > 0.6:  
                person_id = class_id  

                if class_name == "Walking":
                    color = (0, 255, 0)  
                    label =  f"{known_names} Walking"
                    email_sent = False  

                elif class_name == "Sitting":
                    color = (255, 255, 0)  
                    label =  "Sitting"

                elif class_name == "Fall Detected":
                    color = (0, 0, 255)  
                    label = "Fallen"
                    print("Fall detected! Checking conditions...")

                    if person_id in previous_y_positions:
                        y_difference = previous_y_positions[person_id] - y1
                        print(f"Y Difference: {y_difference}")  

                        if not email_sent:
                            print(f"Fall detected! Sending email alert for {detected_name}...")
                            alert_data = {"alert": f"{detected_name} Fall Detected!"}
                            print(json.dumps(alert_data))  
                            putTextRect(frame, "Fall Detected!", (x1, y1 - 20), scale=2, thickness=3, colorR=color)

                            
                            winsound.PlaySound("alert.wav", winsound.SND_FILENAME | winsound.SND_ASYNC)
                            if send_email("hashirkp13@gmail.com", f"Fall Detected: {detected_name}", f"{detected_name} has fallen. Immediate attention needed!"):
                                email_sent = True  

                previous_y_positions[person_id] = y1

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                putTextRect(frame, label, (x1, y1 - 10), scale=1, thickness=2, colorR=color)

    cv2.imshow("Fall Detection System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
# import cv2
# import os
# import face_recognition
# import numpy as np
# import pickle
# import json
# from ultralytics import YOLO
# from cvzone.Utils import putTextRect
# import smtplib
# from email.mime.text import MIMEText
# from email.mime.multipart import MIMEMultipart
# import winsound

# # Load face encodings from file
# with open("face_encodings.pkl", "rb") as f:
#     face_data = pickle.load(f)
#     known_face_encodings = face_data["encodings"]
#     known_names = face_data["names"]

# # Email alert function
# def send_email(to_email, subject, message):
#     sender_email = "support@zoftcares.in"  
#     sender_password = "U_Dr*0R($q&?"    

#     msg = MIMEMultipart()
#     msg["From"] = sender_email
#     msg["To"] = to_email
#     msg["Subject"] = subject
#     msg.attach(MIMEText(message, "plain"))

#     try:
#         server = smtplib.SMTP_SSL("mail.zoftcares.in", 465)
#         server.login(sender_email, sender_password)
#         server.sendmail(sender_email, to_email, msg.as_string())
#         server.quit()
#         print("Email sent successfully!")
#         return True
#     except Exception as e:
#         print(f"Email sending failed: {e}")
#         return False

# # Load trained YOLO model
# model = YOLO('C:\\kaggle_incident_detection\\kaggle\\runs\\detect\\train_dataset\\weights\\best.pt')

# # Open video file
# cap = cv2.VideoCapture('hafeefa2_fall.mp4')

# previous_y_positions = {}
# fall_threshold = 130  
# email_sent = False  

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break  

#     # Face recognition first
#     detected_name = ["Unknown",""]  
#     small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # Reduce processing time
#     rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
#     face_locations = face_recognition.face_locations(rgb_frame)
#     face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

#     for face_encoding in face_encodings:
#         matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
#         face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
#         best_match_index = np.argmin(face_distances) if any(matches) else None

#         if best_match_index is not None and matches[best_match_index]:
#             detected_name = known_names[best_match_index]  # Set recognized person's name

#     # Run YOLO inference
#     results = model(frame)

#     for result in results:
#         boxes = result.boxes  

#         for box in boxes:
#             class_id = int(box.cls)  
#             confidence = float(box.conf)  
#             x1, y1, x2, y2 = map(int, box.xyxy[0])  

#             class_name = model.names[class_id]
#             print(f"Detected: {class_name} (Confidence: {confidence:.2f})")  

#             if confidence > 0.6:  
#                 person_id = class_id  

#                 if class_name == "Walking":
#                     color = (0, 255, 0)  
#                     label = "Walking"
#                     email_sent = False  

#                 elif class_name == "Sitting":
#                     color = (255, 255, 0)  
#                     label = "Sitting"

#                 elif class_name == "Fall Detected":
#                     color = (0, 0, 255)  
#                     label = "Fallen"
#                     print("Fall detected! Checking conditions...")

#                     if person_id in previous_y_positions:
#                         y_difference = previous_y_positions[person_id] - y1
#                         print(f"Y Difference: {y_difference}")  

#                         if not email_sent:
#                             print(f"Fall detected! Sending email alert for {detected_name}...")
#                             alert_data = {"alert": f"{detected_name} Fall Detected!"}
#                             print(json.dumps(alert_data))  
#                             putTextRect(frame, "Fall Detected!", (x1, y1 - 20), scale=2, thickness=3, colorR=color)

#                             # Play alert sound
#                             winsound.PlaySound("alert.wav", winsound.SND_FILENAME | winsound.SND_ASYNC)

#                             # Send email with detected name
#                             if send_email("hashirkp13@gmail.com", f"Fall Detected: {detected_name}", f"{detected_name} has fallen. Immediate attention needed!"):
#                                 email_sent = True  

#                 previous_y_positions[person_id] = y1

#                 # Draw bounding box and label
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
#                 putTextRect(frame, label, (x1, y1 - 10), scale=1, thickness=2, colorR=color)

#     cv2.imshow("Fall Detection System", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
#-----------------------------------
