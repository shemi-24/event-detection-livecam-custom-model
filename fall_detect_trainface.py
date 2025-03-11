# this code njn  matte main code cheythathan
# import cv2
# import pickle
# import json
# import numpy as np
# from ultralytics import YOLO
# from cvzone.Utils import putTextRect
# import smtplib
# from email.mime.text import MIMEText
# from email.mime.multipart import MIMEMultipart
# import os
# import face_recognition

# # Load known face embeddings
# with open("C:\\kaggle_incident_detection\\kaggle\\face_embeddings.pkl", "rb") as f:
#     face_data = pickle.load(f)

# known_face_encodings = np.array(face_data["embeddings"])  # Ensure it's an array
# known_face_names = face_data["labels"]

# # Fix shape mismatch (Ensure embeddings have 128 dimensions)
# if known_face_encodings.shape[1] != 128:
#     print(f"Warning: Face embeddings have {known_face_encodings.shape[1]} dimensions instead of 128!")
#     known_face_encodings = known_face_encodings[:, :128]  # Trim to first 128 dimensions

# # Email alert function
# def send_email(to_email, subject, message):
#     sender_email = "support@zoftcares.in"  # Replace with your email
#     sender_password = "U_Dr*0R($q&?"        # Replace with your email password

#     msg = MIMEMultipart()
#     msg["From"] = sender_email
#     msg["To"] = to_email
#     msg["Subject"] = subject
#     msg.attach(MIMEText(message, "plain"))

#     try:
#         server = smtplib.SMTP_SSL("mail.zoftcares.in", 465)  # Use SSL instead of SMTP
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
# RTSP_URL = 'rtsp://rahil:Rahil123@112.133.238.189:554/Streaming/channels/301'

# os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'


# cap = cv2.VideoCapture(RTSP_URL,cv2.CAP_FFMPEG)

# previous_y_positions = {}
# fall_threshold = 130  # Adjust as needed
# email_sent = False

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break  # Exit if no frame

#     # Convert frame to RGB for face recognition
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     # Run YOLO inference
#     results = model(frame)

#     for result in results:
#         boxes = result.boxes  # Get detected objects

#         for box in boxes:
#             class_id = int(box.cls)  # Get class ID
#             confidence = float(box.conf)  # Get confidence score
#             x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates

#             # Get class name
#             class_name = model.names[class_id]
#             print(f"Detected: {class_name} (Confidence: {confidence:.2f})")  # Debugging

#             if confidence > 0.6:
#                 person_id = class_id  # Assuming only one person detected at a time

#                 if class_name == "Walking":
#                     color = (0, 255, 0)
#                     label = "Person Walking"
#                     email_sent = False
#                 elif class_name == "Sitting":
#                     color = (255, 255, 0)
#                     label = "Person Sitting"
#                 elif class_name == "Fall Detected":
#                     color = (0, 0, 255)
#                     label = "Person Fallen"
#                     print("Fall detected! Checking conditions...")

#                     if person_id in previous_y_positions:
#                         y_difference = previous_y_positions[person_id] - y1
#                         print(f"Y Difference: {y_difference}")  

#                         if not email_sent:
#                             print("Fall detected! Sending email alert...")
#                             alert_data = {"alert": "Person Fall Detected!"}
#                             print(json.dumps(alert_data))  # Print JSON alert
#                             putTextRect(frame, "Fall Detected!", (x1, y1 - 20), scale=2, thickness=3, colorR=color)

#                             if send_email("hashirkp13@gmail.com", "Fall Detected!", "A person has fallen. Immediate attention needed!"):
#                                 email_sent = True  

#                 previous_y_positions[person_id] = y1

#                 # Draw bounding box and label using CVZone
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
#                 putTextRect(frame, label, (x1, y1 - 10), scale=1, thickness=2, colorR=color)

#     # **Face Recognition**
#     face_locations = face_recognition.face_locations(rgb_frame)
#     face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

#     for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
#         # Ensure face encoding is 128 dimensions
#         if face_encoding.shape[0] != 128:
#             print("Face encoding shape mismatch! Skipping face.")
#             continue  # Skip if not 128 dimensions

#         matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.7)
#         name = "Unknown"

#         face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
#         best_match_index = np.argmin(face_distances)

#         if matches[best_match_index]:
#             name = known_face_names[best_match_index]

#         # Draw a rectangle around the face
#         cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 2)
#         putTextRect(frame, name, (left, top - 10), scale=1, thickness=2, colorR=(0, 255, 255))

#     # Show video with detections
#     cv2.imshow("Fall & Face Detection System", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()

#---- ee code only recognize my video not hafeefa
import cv2
import json
import pickle
import smtplib
import face_recognition
from ultralytics import YOLO
from cvzone.Utils import putTextRect
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Load known face encodings
with open("C:\\kaggle_incident_detection\\kaggle\\face_embeddings.pkl", "rb") as f:
    face_data = pickle.load(f)

known_face_encodings = face_data["embeddings"]
known_face_names = face_data["labels"]

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
        server = smtplib.SMTP_SSL("mail.zoftcares.in", 465)  
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
cap = cv2.VideoCapture('hashir_fall.mp4')

previous_y_positions = {}
fall_threshold = 130  
email_sent = False  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Face detection
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_labels = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
        name = "Unknown"

        if True in matches:
            matched_index = matches.index(True)
            name = known_face_names[matched_index]

        face_labels.append(name)

    # Run YOLO inference
    results = model(frame)

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
                    label = "Person Walking"
                    email_sent = False  
                elif class_name == "Sitting":
                    color = (255, 255, 0)  
                    label = "Person Sitting"
                elif class_name == "Fall Detected":
                    color = (0, 0, 255)  
                    label = "Person Fallen"
                    print("Fall detected! Checking conditions...")

                    if person_id in previous_y_positions:
                        y_difference = previous_y_positions[person_id] - y1
                        print(f"Y Difference: {y_difference}")  

                        if not email_sent:
                            print("Fall detected! Sending email alert...")
                            alert_data = {"alert": "Person Fall Detected!"}
                            print(json.dumps(alert_data))  
                            putTextRect(frame, "Fall Detected!", (x1, y1 - 20), scale=2, thickness=3, colorR=color)

                            if send_email("hashirkp13@gmail.com", "Fall Detected!", "A person has fallen. Immediate attention needed!"):
                                email_sent = True  

                previous_y_positions[person_id] = y1

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                putTextRect(frame, label, (x1, y1 - 10), scale=1, thickness=2, colorR=color)

    # Draw face recognition results
    for (top, right, bottom, left), name in zip(face_locations, face_labels):
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 255), 2)
        putTextRect(frame, name, (left, top - 10), scale=1, thickness=2, colorR=(255, 0, 255))

    # Show video
    cv2.imshow("Incident Detection with Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

#-----------------------------------
# import cv2
# import json
# from ultralytics import YOLO
# from cvzone.Utils import putTextRect
# import smtplib
# from email.mime.text import MIMEText
# from email.mime.multipart import MIMEMultipart

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

# # Load trained YOLO models
# face_model = YOLO('C:\\kaggle_incident_detection\\kaggle\\data-trained-yolo2\\weights\\best.pt')  # Face recognition model
# detection_model = YOLO('C:\\kaggle_incident_detection\\kaggle\\runs\\detect\\train_dataset\\weights\\best.pt')  # Fall detection model

# # Open video file
# cap = cv2.VideoCapture('hafeefa2_fall.mp4')

# previous_y_positions = {}
# fall_threshold = 130  
# email_sent = False

# last_detected_person = "Unknown"
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     detected_person = "Unknown"
    
#     # Face detection and recognition using YOLO
#     face_results = face_model(frame)
#     for face_result in face_results:
#         for box in face_result.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             confidence = float(box.conf)
#             class_id = int(box.cls)
#             person_name = face_model.names[class_id]  # Get person's name from model labels
            
#             if confidence > 0.6:  # Confidence threshold for face recognition
#                 detected_person = person_name
#                 last_detected_person = detected_person
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(frame, detected_person, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
#     # YOLO Fall Detection
#     results = detection_model(frame)
#     for result in results:
#         for box in result.boxes:
#             class_id = int(box.cls)
#             confidence = float(box.conf)
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             class_name = detection_model.names[class_id]

#             if confidence > 0.6:
#                 person_id = class_id
                
#                 if class_name == "Walking":
#                     color = (0, 255, 0)
#                     label = "Person Walking"
#                     email_sent = False
#                 elif class_name == "Sitting":
#                     color = (255, 255, 0)
#                     label = "Person Sitting"
#                 elif class_name == "Fall Detected":
#                     color = (0, 0, 255)
                    
#                     if detected_person == "Unknown":
#                         detected_person = last_detected_person

#                     label = f"Person Fallen ({detected_person})"
#                     print("Fall detected! Checking conditions...")
                    
#                     if person_id in previous_y_positions:
#                         y_difference = previous_y_positions[person_id] - y1
#                         print(f"Y Difference: {y_difference}")

#                         if not email_sent:
#                             print(f"Fall detected! Sending email alert for {detected_person}...")
#                             alert_data = {"alert": f"Fall Detected for {detected_person}"}
#                             print(json.dumps(alert_data))
#                             putTextRect(frame, f"Fall Detected! ({detected_person})", (x1, y1 - 20), scale=2, thickness=3, colorR=color)

#                             email_body = f"{detected_person} has fallen. Immediate attention needed!"
#                             if send_email("hashirkp13@gmail.com", "Fall Detected!", email_body):
#                                 email_sent = True

#                 previous_y_positions[person_id] = y1
                
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
#                 putTextRect(frame, label, (x1, y1 - 10), scale=1, thickness=2, colorR=color)

#     cv2.imshow("Fall Detection System with Face Recognition", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

#-----first code
# import cv2
# import json
# from ultralytics import YOLO
# from cvzone.Utils import putTextRect
# import smtplib
# from email.mime.text import MIMEText
# from email.mime.multipart import MIMEMultipart

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

# # Load trained YOLO model for face and fall detection
# face_model = YOLO('C:\kaggle_incident_detection\kaggle\data-trained-yolo2\\weights\\best.pt')  # Face recognition model
# detection_model = YOLO('C:\\kaggle_incident_detection\\kaggle\\runs\\detect\\train_dataset\\weights\\best.pt')  # Fall detection model

# # Open video file
# cap = cv2.VideoCapture('hafeefa2_fall.mp4')

# previous_y_positions = {}
# fall_threshold = 130  
# email_sent = False

# last_detected_person = "Unknown"
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     detected_person = "Unknown"
    
#     # Face recognition using YOLO
#     face_results = face_model(frame)
#     for face_result in face_results:
#         for box in face_result.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             confidence = float(box.conf)
#             class_id = int(box.cls)
#             person_name = face_model.names[class_id]  # Get person's name from model labels
            
#             if confidence > 0.6:
#                 detected_person = person_name
#                 last_detected_person = detected_person
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(frame, detected_person, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
#     # YOLO Fall Detection
#     results = detection_model(frame)
#     for result in results:
#         for box in result.boxes:
#             class_id = int(box.cls)
#             confidence = float(box.conf)
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             class_name = detection_model.names[class_id]

#             if confidence > 0.6:
#                 person_id = class_id
                
#                 if class_name == "Walking":
#                     color = (0, 255, 0)
#                     label = "Person Walking"
#                     email_sent = False
#                 elif class_name == "Sitting":
#                     color = (255, 255, 0)
#                     label = "Person Sitting"
#                 elif class_name == "Fall Detected":
#                     color = (0, 0, 255)
                    
#                     if detected_person == "Unknown":
#                         detected_person = last_detected_person

#                     label = f"Person Fallen ({detected_person})"
#                     print("Fall detected! Checking conditions...")
                    
#                     if person_id in previous_y_positions:
#                         y_difference = previous_y_positions[person_id] - y1
#                         print(f"Y Difference: {y_difference}")

#                         if not email_sent:
#                             print(f"Fall detected! Sending email alert for {detected_person}...")
#                             alert_data = {"alert": f"Fall Detected for {detected_person}"}
#                             print(json.dumps(alert_data))
#                             putTextRect(frame, f"Fall Detected! ({detected_person})", (x1, y1 - 20), scale=2, thickness=3, colorR=color)

#                             email_body = f"{detected_person} has fallen. Immediate attention needed!"
#                             if send_email("hashirkp13@gmail.com", "Fall Detected!", email_body):
#                                 email_sent = True

#                 previous_y_positions[person_id] = y1
                
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
#                 putTextRect(frame, label, (x1, y1 - 10), scale=1, thickness=2, colorR=color)

#     cv2.imshow("Fall Detection System with Face Recognition", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()