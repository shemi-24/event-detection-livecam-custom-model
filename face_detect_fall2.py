
# # this code is the latest code of my face_recognize2.py and incident_detection (testfile.py)- integration
import cv2
import face_recognition
import json
from ultralytics import YOLO
from cvzone.Utils import putTextRect
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

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

# Load trained YOLO model for fall detection
model = YOLO('C:\\kaggle_incident_detection\\kaggle\\runs\\detect\\train_dataset\\weights\\best.pt')

# Load known faces
known_face_encodings = []
known_face_names = []

# Load images & encode faces
known_person1_img = face_recognition.load_image_file('img10.jpg')  # Hafeefa image
known_person2_img = face_recognition.load_image_file('img18.jpg')  # Shameer image

known_person1_encoding = face_recognition.face_encodings(known_person1_img)[0]
known_person2_encoding = face_recognition.face_encodings(known_person2_img)[0]

known_face_encodings.append(known_person1_encoding)
known_face_encodings.append(known_person2_encoding)

known_face_names.append('Hafeefa')
known_face_names.append('Shameer')

# Open video file
cap = cv2.VideoCapture('hafeefa2_fall.mp4')

previous_y_positions = {}
fall_threshold = 130  
email_sent = False

last_detected_person="Unknown"
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB for face recognition
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Face recognition
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    detected_person = "Unknown"

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        
        if True in matches:
            first_match_index = matches.index(True)
            detected_person = known_face_names[first_match_index]
            last_detected_person = detected_person

        # Draw face recognition box
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, detected_person, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # YOLO Fall Detection
    results = model(frame)

    for result in results:
        boxes = result.boxes  

        for box in boxes:
            class_id = int(box.cls)  
            confidence = float(box.conf)  
            x1, y1, x2, y2 = map(int, box.xyxy[0])  

            class_name = model.names[class_id]

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

                    if detected_person is "Unknown":
                        detected_person=last_detected_person

                    label = f"Person Fallen ({detected_person})"
                    print("Fall detected! Checking conditions...")

                    if person_id in previous_y_positions:
                        y_difference = previous_y_positions[person_id] - y1
                        print(f"Y Difference: {y_difference}")  

                        if not email_sent:
                            print(f"Fall detected! Sending email alert for {detected_person}...")
                            alert_data = {"alert": f"Fall Detected for {detected_person}"}
                            print(json.dumps(alert_data))

                            putTextRect(frame, f"Fall Detected! ({detected_person})", (x1, y1 - 20), scale=2, thickness=3, colorR=color)

                            # Send email only once per fall event
                            email_body = f"{detected_person} has fallen. Immediate attention needed!"
                            if send_email("hashirkp13@gmail.com", "Fall Detected!", email_body):
                                email_sent = True  

                previous_y_positions[person_id] = y1

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                putTextRect(frame, label, (x1, y1 - 10), scale=1, thickness=2, colorR=color)

    cv2.imshow("Fall Detection System with Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# import cv2
# import json
# from ultralytics import YOLO
# from cvzone.Utils import putTextRect
# import smtplib
# from email.mime.text import MIMEText
# from email.mime.multipart import MIMEMultipart
# import face_recognition

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

# # Load known faces for recognition
# known_face_encoding = []
# known_face_names = []

# # Load and encode known faces
# known_person1_img = face_recognition.load_image_file('img10.jpg')  # Hafeefa image
# known_person2_img = face_recognition.load_image_file('img18.jpg')  # Shameer image

# known_person1_encoding = face_recognition.face_encodings(known_person1_img)[0]
# known_person2_encoding = face_recognition.face_encodings(known_person2_img)[0]

# known_face_encoding.append(known_person1_encoding)
# known_face_encoding.append(known_person2_encoding)

# known_face_names.append('Hafeefa')
# known_face_names.append('Shameer')

# # Open video file
# cap = cv2.VideoCapture('hafeefa2_fall.mp4')
# if not cap.isOpened():
#     print("Error: Could not open video file.")
#     exit()

# # Track previous Y position for fall detection
# previous_y_positions = {}
# fall_threshold = 130  
# email_sent = False  

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break  

#     # Convert frame to RGB for face recognition
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     # Face recognition
#     face_locations = face_recognition.face_locations(rgb_frame)
#     face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

#     # Recognize faces
#     recognized_names = []
#     for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
#         matches = face_recognition.compare_faces(known_face_encoding, face_encoding)
#         name = "Unknown"

#         if True in matches:
#             first_match_index = matches.index(True)
#             name = known_face_names[first_match_index]

#         recognized_names.append(name)
#         cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
#         cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

#     # Run YOLO inference for fall detection
#     results = model(frame)

#     for result in results:
#         boxes = result.boxes  # Get detected objects

#         for box in boxes:
#             class_id = int(box.cls)  # Get class ID
#             confidence = float(box.conf)  # Get confidence score
#             x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates

#             # Get class name
#             class_name = model.names[class_id]
#             print(f"Detected: {class_name} (Confidence: {confidence:.2f})")  

#             if confidence > 0.6:  # Confidence threshold
#                 person_id = class_id  # Assuming only one person detected at a time

#                 if class_name == "Fall Detected":
#                     print("Fall detected! Checking conditions...")

#                     # Track Y position to detect actual fall
#                     if person_id in previous_y_positions:
#                         y_difference = previous_y_positions[person_id] - y1
#                         print(f"Y Difference: {y_difference}")  # Debugging

#                         if not email_sent:
#                             print("Fall detected! Sending email alert...")
#                             alert_data = {"alert": "Person Fall Detected!"}
#                             print(json.dumps(alert_data))  
#                             putTextRect(frame, "Fall Detected!", (x1, y1 - 20), scale=2, thickness=3, colorR=(0, 0, 255))

#                             # Get the recognized name (if any)
#                             if recognized_names:
#                                 person_name = recognized_names[0]  # Assume the first recognized person
#                             else:
#                                 person_name = "Unknown Person"

#                             # Send email with the recognized person's name
#                             if send_email("hashirkp13@gmail.com", "Fall Detected!", f"{person_name} has fallen. Immediate attention needed!"):
#                                 email_sent = True  

#                 # Update previous Y position
#                 previous_y_positions[person_id] = y1

#                 # Draw bounding box and label using CVZone
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
#                 putTextRect(frame, class_name, (x1, y1 - 10), scale=1, thickness=2, colorR=(0, 0, 255))

#     # Show video with detections
#     cv2.imshow("Fall Detection System with Face Recognition", frame)

#     # Exit on 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()
