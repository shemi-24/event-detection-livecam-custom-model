# import cv2
# import json
# import smtplib
# import threading
# from email.mime.text import MIMEText
# from email.mime.multipart import MIMEMultipart
# from ultralytics import YOLO
# import mediapipe as mp

# # Load trained YOLO model
# model = YOLO('C:\\kaggle_incident_detection\\kaggle\\runs\\detect\\train_dataset\\weights\\best.pt')

# # Initialize MediaPipe Pose Estimation
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.7)

# # SMTP Server Settings
# SMTP_SERVER = "mail.zoftcares.in"
# SMTP_PORT = 465
# EMAIL_SENDER = "support@zoftcares.in"
# EMAIL_PASSWORD = "U_Dr*OR($q&?"  # Use App Password if needed
# EMAIL_RECEIVER = "hashirkp90@gmail.com"

# # Function to send email alert
# def send_email_alert():
#     msg = MIMEMultipart()
#     msg["From"] = EMAIL_SENDER
#     msg["To"] = EMAIL_RECEIVER
#     msg["Subject"] = "🚨 Fall Detected Alert!"

#     body = "ALERT: A person has fallen! Immediate attention required."
#     msg.attach(MIMEText(body, "plain"))

#     try:
#         server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
#         server.starttls()
#         server.login(EMAIL_SENDER, EMAIL_PASSWORD)
#         server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
#         server.quit()
#         print("✅ Email Sent Successfully!")
#     except Exception as e:
#         print(f"❌ Error: {e}")

# # Open webcam
# cap = cv2.VideoCapture(0)

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break  # Exit if no frame

#     object_detected = False

#     # Run YOLOv8 inference
#     results = model(frame)

#     for result in results:
#         boxes = result.boxes  # Get detected objects

#         for box in boxes:
#             class_id = int(box.cls)
#             confidence = float(box.conf)
#             x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
#             class_name = model.names[class_id]

#             if confidence > 0.8898:
#                 object_detected = True

#                 if class_name == "Walking":
#                     color = (0, 255, 0)
#                     label = "Person Walking"
#                 elif class_name == "Sitting":
#                     color = (0, 255, 0)
#                     label = "Person Sitting"
#                 elif class_name == "Fall Detected":
#                     color = (0, 0, 255)
#                     label = "🚨 Person Fallen!"

#                     # Trigger fall alert (email)
#                     print("⚠️ Fall detected! Sending alert email...")
#                     threading.Thread(target=send_email_alert).start()

#                 # Draw bounding box
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
#                 cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

#     # Run MediaPipe Pose Estimation if object detected
#     if object_detected:
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         pose.process(rgb_frame)

#     # Show video with detections
#     cv2.imshow("Fall Detection System", frame)

#     # Exit on 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()

# first cheythath-------------------------------------------
# import cv2
# import json
# from ultralytics import YOLO
# import mediapipe as mp

# # Load trained YOLO model
# model = YOLO('C:\\kaggle_incident_detection\\kaggle\\runs\detect\\train_dataset\\weights\\best.pt')

# # Initialize MediaPipe Pose Estimation
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.7)

# # Open webcam
# cap = cv2.VideoCapture(0)

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break  # Exit if no frame

#     alert_data = None  # Default: No alert
#     object_detected = False  # Track if any object is detected

#     # Run YOLOv8 inference
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

#             if confidence > 0.8898:  # Confidence threshold
#                 object_detected = True  # Mark object as detected

#                 if class_name == "Walking":
#                     color = (0, 255, 0)  # Green for sitting
#                     label = "person walking"  # Label for sitting
#                 elif class_name=="Sitting":
#                     color=(0,255,0)
#                     label="person sitting"    
                
#                 elif class_name == "Fall Detected":
#                     color = (0, 0, 255)  # Red for fall
#                     label = "person fall"  # Label for fall

#                     # Trigger fall alert
#                     alert_data = {"alert": "Person Fall Detected!"}
#                     print(json.dumps(alert_data))  # Print JSON alert

#                 # Draw bounding box and label
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
#                 cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

#     # Run MediaPipe Pose Estimation if any object is detected
#     if object_detected:
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         pose_results = pose.process(rgb_frame)

#     # Show video with detections
#     cv2.imshow("Fall Detection System", frame)

#     # Exit on 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()
#----------------currently last corect aayath
# import cv2
# import json
# from ultralytics import YOLO
# import mediapipe as mp

# # Load trained YOLO model
# model = YOLO('C:\\kaggle_incident_detection\\kaggle\\runs\\detect\\train_dataset\\weights\\best.pt')

# # Initialize MediaPipe Pose Estimation
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.7)

# # Open webcam
# cap = cv2.VideoCapture('safa_fall.mp4')

# # Track previous Y position for fall detection
# previous_y_positions = {}
# fall_threshold = 125  # Adjust this threshold as needed

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break  # Exit if no frame

#     alert_data = None  # Default: No alert
#     object_detected = False  # Track if any object is detected

#     # Run YOLOv8 inference
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

#             if confidence > 0.7:  # Confidence threshold
#                 object_detected = True  # Mark object as detected
#                 person_id = class_id  # Assuming only one person detected at a time

#                 if class_name == "Walking":
#                     color = (0, 255, 0)  # Green for walking
#                     label = "Person Walking"
#                 elif class_name == "Sitting":
#                     color = (255, 255, 0)  # Yellow for sitting
#                     label = "Person Sitting"
#                 elif class_name == "Fall Detected":
#                     color = (0, 0, 255)  # Red for fall
#                     label = "Person Fallen"

#                     # Track Y position to detect actual fall
#                     if person_id in previous_y_positions:
#                         y_difference = previous_y_positions[person_id] - y1
#                         if y_difference > fall_threshold:
#                             alert_data = {"alert": "Person Fall Detected!"}
#                             print(json.dumps(alert_data))  # Print JSON alert
                    
#                 # Update previous Y position
#                 previous_y_positions[person_id] = y1

#                 # Draw bounding box and label
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
#                 cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

#     # Run MediaPipe Pose Estimation if any object is detected
#     if object_detected:
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         pose_results = pose.process(rgb_frame)

#     # Show video with detections
#     cv2.imshow("Fall Detection System", frame)

#     # Exit on 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()
#---------------------------------------- last vs code copy paste aakyapo korch rdy ayath
# import cv2
# import json
# import math
# from ultralytics import YOLO
# import mediapipe as mp

# # Load trained YOLO model
# model = YOLO('C:\\kaggle_incident_detection\\kaggle\\runs\\detect\\train_dataset\\weights\\best.pt')

# # Initialize MediaPipe Pose Estimation
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.7)

# # Open webcam
# cap = cv2.VideoCapture(0)

# # Track previous Y position for fall detection
# previous_y_positions = {}
# fall_threshold = 100  # Adjust as needed

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break  # Exit if no frame
    
#     frame = cv2.resize(frame, (640, 480))  # Resize frame for consistency
#     alert_data = None  # Default: No alert
#     object_detected = False  # Track detected objects
    
#     # Run YOLOv8 inference
#     results = model(frame)
    
#     for result in results:
#         boxes = result.boxes  # Get detected objects
#         for box in boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
#             confidence = float(box.conf[0])  # Confidence score
#             class_detect = int(box.cls[0])  # Class ID
#             class_name = model.names[class_detect]  # Get class name
            
#             conf = math.ceil(confidence * 100)  # Convert confidence to percentage
#             height = y2 - y1
#             width = x2 - x1
#             threshold = height - width  # Key metric for fall detection
            
#             if conf > 70 and class_name == 'person':
#                 object_detected = True
                
#                 # Draw bounding box and label
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
#                 cv2.putText(frame, class_name, (x1 + 8, y1 - 12), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
#                 # Fall detection logic
#                 if threshold < 0:  # Condition for lying down
#                     if class_detect in previous_y_positions:
#                         y_difference = previous_y_positions[class_detect] - y1
#                         if y_difference > fall_threshold:
#                             alert_data = {"alert": "Person Fall Detected!"}
#                             print(json.dumps(alert_data))
#                             cv2.putText(frame, "Fall Detected", (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
#                 # Update previous Y position
#                 previous_y_positions[class_detect] = y1
    
#     # Run MediaPipe Pose Estimation if a person is detected
#     if object_detected:
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         pose_results = pose.process(rgb_frame)
    
#     # Show video with detections
#     cv2.imshow("Fall Detection System", frame)
    
#     # Exit on 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()
#-------------------------------ith ipo last fall detected maathram aayath
# import cv2
# import math
# from ultralytics import YOLO
# import cvzone

# # Load trained YOLO model
# model = YOLO('C:\\kaggle_incident_detection\\kaggle\\runs\\detect\\train_dataset\\weights\\best.pt')

# # Open webcam
# cap = cv2.VideoCapture(0)

# # Fall detection threshold
# fall_threshold = 50  # Adjust this threshold as needed (negative value indicates fall)

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break  # Exit if no frame

#     # Run YOLOv8 inference
#     results = model(frame)

#     for result in results:
#         boxes = result.boxes  # Get detected objects

#         for box in boxes:
#             class_id = int(box.cls)  # Get class ID
#             confidence = float(box.conf)  # Get confidence score
#             x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates

#             # Get class name
#             class_name = model.names[class_id]
#             conf_percent = math.ceil(confidence * 100)  # Confidence percentage

#             # Only process if confidence is above 70% and class is 'person'
#             if conf_percent > 70 and class_name == "Fall Detected":
#                 height = y2 - y1  # Height of the bounding box
#                 width = x2 - x1  # Width of the bounding box
#                 threshold = height - width  # Fall detection logic

#                 # Draw bounding box and label using cvzone
#                 cvzone.cornerRect(frame, [x1, y1, width, height], l=30, rt=6)
#                 cvzone.putTextRect(frame, f'{class_name} {conf_percent}%', [x1 + 8, y1 - 12], thickness=1, scale=1)

#                 # Fall detection logic
#                 if threshold < fall_threshold:
#                     cvzone.putTextRect(frame, 'Fall Detected', [x1, y1 - 50], thickness=2, scale=2, colorR=(0, 0, 255))
#                     print("Alert: Person Fall Detected!")

#     # Show video with detections
#     cv2.imshow("Fall Detection System", frame)

#     # Exit on 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()
#----------------------------demo okke crct aayit setayath latest corect code-------------------------
# import cv2
# import json
# from ultralytics import YOLO
# from cvzone.Utils import putTextRect

# # Load trained YOLO model
# model = YOLO('C:\\kaggle_incident_detection\\kaggle\\runs\\detect\\train_dataset\\weights\\best.pt')

# # Open webcam
# cap = cv2.VideoCapture('hashir_fall.mp4')

# # Track previous Y position for fall detection
# previous_y_positions = {}
# fall_threshold = 130  # Adjust this threshold as needed

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break  # Exit if no frame

#     alert_data = None  # Default: No alert
#     object_detected = False  # Track if any object is detected

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

#             if confidence > 0.6:  # Confidence threshold
#                 object_detected = True  # Mark object as detected
#                 person_id = class_id  # Assuming only one person detected at a time

#                 if class_name == "Walking":
#                     color = (0, 255, 0)  # Green for walking
#                     label = "Person Walking"
#                 elif class_name == "Sitting":
#                     color = (255, 255, 0)  # Yellow for sitting
#                     label = "Person Sitting"
#                 elif class_name == "Fall Detected":
#                     color = (0, 0, 255)  # Red for fall
#                     label = "Person Fallen"

#                     # Track Y position to detect actual fall
#                     if person_id in previous_y_positions:
#                         y_difference = previous_y_positions[person_id] - y1
#                         if y_difference > fall_threshold:
#                             alert_data = {"alert": "Person Fall Detected!"}
#                             print(json.dumps(alert_data))  # Print JSON alert
#                             putTextRect(frame, "Fall Detected!", (x1, y1 - 20), scale=2, thickness=3, colorR=color)
                
#                 # Update previous Y position
#                 previous_y_positions[person_id] = y1

#                 # Draw bounding box and label using CVZone
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
#                 putTextRect(frame, label, (x1, y1 - 10), scale=1, thickness=2, colorR=color)

#     # Show video with detections
#     cv2.imshow("Fall Detection System", frame)

#     # Exit on 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()

#-------------------------------------

# import cv2
# import json
# from ultralytics import YOLO
# from cvzone import putTextRect

# # Load trained YOLO model
# model = YOLO('C:\\kaggle_incident_detection\\kaggle\\runs\\detect\\train_dataset\\weights\\best.pt')

# # Open webcam or video
# cap = cv2.VideoCapture('safa_fall.mp4')  # Replace with your video path or 0 for webcam

# # Track previous Y position for fall detection
# previous_y_positions = {}
# fall_threshold = 125  # Adjust this threshold as needed

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break  # Exit if no frame

#     alert_data = None  # Default: No alert

#     # Run YOLOv8 inference
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

#             if confidence > 0.7:  # Confidence threshold
#                 person_id = class_id  # Assuming only one person detected at a time

#                 if class_name == "Walking":
#                     color = (0, 255, 0)  # Green for walking
#                     label = "Person Walking"
#                 elif class_name == "Sitting":
#                     color = (255, 255, 0)  # Yellow for sitting
#                     label = "Person Sitting"
#                 elif class_name == "Fall Detected":
#                     color = (0, 0, 255)  # Red for fall
#                     label = "Person Fallen"

#                     # Track Y position to detect actual fall
#                     if person_id in previous_y_positions:
#                         y_difference = previous_y_positions[person_id] - y1
#                         if y_difference > fall_threshold:
#                             alert_data = {"alert": "Person Fall Detected!"}
#                             print(json.dumps(alert_data))  # Print JSON alert

#                 # Update previous Y position
#                 previous_y_positions[person_id] = y1

#                 # Draw bounding box and label using cvzone
#                 putTextRect(frame, label, (x1, y1 - 30), scale=1, thickness=2, colorT=color, colorR=(0, 0, 0))
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

#     # Show video with detections
#     cv2.imshow("Fall Detection System", frame)

#     # Exit on 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()
# import cv2
# import json
# import smtplib
# from email.mime.text import MIMEText
# from email.mime.multipart import MIMEMultipart
# from ultralytics import YOLO
# from cvzone.Utils import putTextRect

# # Load trained YOLO model
# model = YOLO('C:\\kaggle_incident_detection\\kaggle\\runs\\detect\\train_dataset\\weights\\best.pt')

# # Open webcam or video file
# cap = cv2.VideoCapture('hashir_fall.mp4')

# # Track previous Y position for fall detection
# previous_y_positions = {}
# fall_threshold = 130  # Adjust this threshold as needed

# # Email Alert Configuration
# EMAIL_SENDER = "support@zoftcares.in"
# EMAIL_PASSWORD = "U_Dr*0R($q&?"  # Use App Password for security
# EMAIL_RECEIVER = "hashirkp90@gmail.com"

# def send_email_alert():
#     """Function to send an email alert when a fall is detected."""
#     try:
#         subject = "⚠️ Fall Detected Alert!"
#         body = "A fall has been detected in the monitoring system. Immediate attention is required!"

#         # Set up email message
#         msg = MIMEMultipart()
#         msg['From'] = EMAIL_SENDER
#         msg['To'] = EMAIL_RECEIVER
#         msg['Subject'] = subject
#         msg.attach(MIMEText(body, 'plain'))

#         # Connect to the SMTP server
#         server = smtplib.SMTP("mail.zoftcares.in", 465)
#         server.starttls()  # Secure the connection
#         server.login(EMAIL_SENDER, EMAIL_PASSWORD)  # Login to email
#         server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())  # Send email
#         server.quit()
#         print("📧 Email alert sent successfully!")
    
#     except Exception as e:
#         print(f"❌ Email sending failed: {e}")

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break  # Exit if no frame

#     alert_data = None  # Default: No alert
#     object_detected = False  # Track if any object is detected

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

#             if confidence > 0.6:  # Confidence threshold
#                 object_detected = True  # Mark object as detected
#                 person_id = class_id  # Assuming only one person detected at a time

#                 if class_name == "Walking":
#                     color = (0, 255, 0)  # Green for walking
#                     label = "Person Walking"
#                 elif class_name == "Sitting":
#                     color = (255, 255, 0)  # Yellow for sitting
#                     label = "Person Sitting"
#                 elif class_name == "Fall Detected":
#                     color = (0, 0, 255)  # Red for fall
#                     label = "Person Fallen"

#                     # Track Y position to detect actual fall
#                     if person_id in previous_y_positions:
#                         y_difference = previous_y_positions[person_id] - y1
#                         if y_difference > fall_threshold:
#                             alert_data = {"alert": "Person Fall Detected!"}
#                             print(json.dumps(alert_data))  # Print JSON alert
#                             putTextRect(frame, "Fall Detected!", (x1, y1 - 20), scale=2, thickness=3, colorR=color)

#                             # Send Email Alert
#                             send_email_alert()
                
#                 # Update previous Y position
#                 previous_y_positions[person_id] = y1

#                 # Draw bounding box and label using CVZone
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
#                 putTextRect(frame, label, (x1, y1 - 10), scale=1, thickness=2, colorR=color)

#     # Show video with detections
#     cv2.imshow("Fall Detection System", frame)

#     # Exit on 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()
# import cv2
# import json
# import smtplib
# import ssl
# from email.mime.text import MIMEText
# from email.mime.multipart import MIMEMultipart
# from ultralytics import YOLO
# from cvzone.Utils import putTextRect

# # Load trained YOLO model
# model = YOLO('C:\\kaggle_incident_detection\\kaggle\\runs\\detect\\train_dataset\\weights\\best.pt')

# # Open webcam
# cap = cv2.VideoCapture('hashir_fall.mp4')

# # Track previous Y position for fall detection
# previous_y_positions = {}
# fall_threshold = 130  # Adjust this threshold as needed

# # Email Configuration
# SMTP_SERVER = "mail.zoftcares.in"  # Replace with your company SMTP server
# SMTP_PORT = 465  # Common SMTP port (587 for TLS, 465 for SSL)
# EMAIL_SENDER = "support@zoftcares.in"  # Your company email
# EMAIL_PASSWORD = "U_Dr*0R($q&?"  # Your company email password
# EMAIL_RECEIVER = "hashirkp90@gmail.com"  # Change to the recipient's email

# def send_email_alert():
#     """Function to send an email alert when a fall is detected."""
#     subject = "⚠️ Fall Detected Alert"
#     body = "A person has fallen. Immediate assistance is required."

#     msg = MIMEMultipart()
#     msg["From"] = EMAIL_SENDER
#     msg["To"] = EMAIL_RECEIVER
#     msg["Subject"] = subject
#     msg.attach(MIMEText(body, "plain"))

#     try:
#         context = ssl.create_default_context()
#         with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
#             server.starttls(context=context)  # Secure connection
#             server.login(EMAIL_SENDER, EMAIL_PASSWORD)
#             server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
#         print("✅ Alert email sent successfully!")
#     except Exception as e:
#         print(f"❌ Failed to send email: {e}")

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break  # Exit if no frame

#     alert_data = None  # Default: No alert
#     object_detected = False  # Track if any object is detected

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

#             if confidence > 0.6:  # Confidence threshold
#                 object_detected = True  # Mark object as detected
#                 person_id = class_id  # Assuming only one person detected at a time

#                 if class_name == "Walking":
#                     color = (0, 255, 0)  # Green for walking
#                     label = "Person Walking"
#                 elif class_name == "Sitting":
#                     color = (255, 255, 0)  # Yellow for sitting
#                     label = "Person Sitting"
#                 elif class_name == "Fall Detected":
#                     color = (0, 0, 255)  # Red for fall
#                     label = "Person Fallen"
                    

#                     # Track Y position to detect actual fall
#                     if person_id in previous_y_positions:
#                         y_difference = previous_y_positions[person_id] - y1
#                         if y_difference > fall_threshold:
                            

                            
#                             putTextRect(frame, "Fall Detected!", (x1, y1 - 20), scale=2, thickness=3, colorR=color)
                            

                            
                            

#                 # Update previous Y position
#                 previous_y_positions[person_id] = y1

#                 # Draw bounding box and label using CVZone
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
#                 putTextRect(frame, label, (x1, y1 - 10), scale=1, thickness=2, colorR=color)
#     send_email_alert()
#     # Show video with detections
#     cv2.imshow("Fall Detection System", frame)

#     # Exit on 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()
#----- last nokeenath 
# import cv2
# import json
# from ultralytics import YOLO
# from cvzone.Utils import putTextRect
# import smtplib
# from email.message import EmailMessage
# from email.mime.text import MIMEText
# from email.mime.multipart import MIMEMultipart

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
#         server = smtplib.SMTP("mail.zoftcares.in", 25)  # Change SMTP settings if using another provider
#         # server.starttls()
#         server.login(sender_email, sender_password)
#         server.sendmail(sender_email, to_email, msg.as_string())
#         server.quit()
#         return True
#     except Exception as e:
#         return str(e)
# # Load trained YOLO model
# model = YOLO('C:\\kaggle_incident_detection\\kaggle\\runs\\detect\\train_dataset\\weights\\best.pt')

# # Open video file
# cap = cv2.VideoCapture('hashir_fall.mp4')

# # Track previous Y position for fall detection
# previous_y_positions = {}
# fall_threshold = 130  # Adjust as needed

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break  # Exit if no frame

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

#             if confidence > 0.6:  # Confidence threshold
#                 person_id = class_id  # Assuming only one person detected at a time

#                 if class_name == "Walking":
#                     color = (0, 255, 0)  # Green for walking
#                     label = "Person Walking"
#                 elif class_name == "Sitting":
#                     color = (255, 255, 0)  # Yellow for sitting
#                     label = "Person Sitting"
#                 elif class_name == "Fall Detected":
#                     color = (0, 0, 255)  # Red for fall
#                     label = "Person Fallen"
#                     print("Fall detected! Sending email alert...")

#                     # Track Y position to detect actual fall
#                     if person_id in previous_y_positions:
#                         y_difference = previous_y_positions[person_id] - y1
#                         if y_difference > fall_threshold:
#                             print("Fall detected! Sending email alert...")
#                             alert_data={"alert":"fall detcted"}
#                             print(json.dumps(alert_data))
#                             send_email("hashirkp90@gmail.com", "Fall Detected!", "A person has fallen. Immediate attention needed!")
#                     putTextRect(frame, "Fall Detected!", (x1, y1 - 20), scale=2, thickness=3, colorR=color)

                
#                 # Update previous Y position
#                 previous_y_positions[person_id] = y1

#                 # Draw bounding box and label using CVZone
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
#                 putTextRect(frame, label, (x1, y1 - 10), scale=1, thickness=2, colorR=color)

#     # Show video with detections
#     cv2.imshow("Fall Detection System", frame)

#     # Exit on 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()
#----------------------------------------last but the not get the loop-aproximately good
# import cv2
# import json
# from ultralytics import YOLO
# from cvzone.Utils import putTextRect
# import smtplib
# from email.message import EmailMessage
# from email.mime.text import MIMEText
# from email.mime.multipart import MIMEMultipart

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
#         server = smtplib.SMTP("mail.zoftcares.in", 25)  # Change SMTP settings if using another provider
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
# cap = cv2.VideoCapture('hashir_fall.mp4')

# # Track previous Y position for fall detection
# previous_y_positions = {}
# fall_threshold = 130  # Adjust as needed
# email_sent = False  # Track whether the email has been sent

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break  # Exit if no frame

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

#             if confidence > 0.6:  # Confidence threshold
#                 person_id = class_id  # Assuming only one person detected at a time

#                 if class_name == "Walking":
#                     color = (0, 255, 0)  # Green for walking
#                     label = "Person Walking"
#                 elif class_name == "Sitting":
#                     color = (255, 255, 0)  # Yellow for sitting
#                     label = "Person Sitting"
#                 elif class_name == "Fall Detected":
#                     color = (0, 0, 255)  # Red for fall
#                     label = "Person Fallen"
#                     print("Fall detected! Checking conditions...")

#                     # Track Y position to detect actual fall
#                     if person_id in previous_y_positions:
#                         y_difference = previous_y_positions[person_id] - y1
#                         print(f"Y Difference: {y_difference}")  # Debugging

#                         if y_difference > fall_threshold and  email_sent:
                      
#                           print("Fall detected! Sending email alert...")
#                           alert_data = {"alert": "Person Fall Detected!"}
#                           print(json.dumps(alert_data))  # Print JSON alert
#                           putTextRect(frame, "Fall Detected!", (x1, y1 - 20), scale=2, thickness=3, colorR=color)

#                             # Send email only once
#                           if send_email("hashirkp90@gmail.com", "Fall Detected!", "A person has fallen. Immediate attention needed!"):

#                             email_sent = True

#                 # Update previous Y position
#                 previous_y_positions[person_id] = y1

#                 # Draw bounding box and label using CVZone
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
#                 putTextRect(frame, label, (x1, y1 - 10), scale=1, thickness=2, colorR=color)

#     # Show video with detections
#     cv2.imshow("Fall Detection System", frame)

#     # Exit on 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()
#-------------------------------------like this have not get the email msg------
# import cv2
# import json
# from ultralytics import YOLO
# from cvzone.Utils import putTextRect
# import smtplib
# from email.mime.text import MIMEText
# from email.mime.multipart import MIMEMultipart

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
#         server = smtplib.SMTP("mail.zoftcares.in", 465)  # Change SMTP settings if using another provider
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
# cap = cv2.VideoCapture('hashir_fall.mp4')

# # Track previous Y position for fall detection
# previous_y_positions = {}
# fall_threshold = 130  # Adjust as needed
# email_sent = False  # Track whether the email has been sent

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break  # Exit if no frame

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

#             if confidence > 0.6:  # Confidence threshold
#                 person_id = class_id  # Assuming only one person detected at a time

#                 if class_name == "Walking":
#                     color = (0, 255, 0)  # Green for walking
#                     label = "Person Walking"
#                 elif class_name == "Sitting":
#                     color = (255, 255, 0)  # Yellow for sitting
#                     label = "Person Sitting"
#                 elif class_name == "Fall Detected":
#                     color = (0, 0, 255)  # Red for fall
#                     label = "Person Fallen"
#                     print("Fall detected! Checking conditions...")

#                     # Track Y position to detect actual fall
#                     if person_id in previous_y_positions:
#                         y_difference = previous_y_positions[person_id] - y1
#                         print(f"Y Difference: {y_difference}")  # Debugging

#                         if y_difference > fall_threshold and not email_sent:
#                             print("Fall detected! Sending email alert...")
#                             alert_data = {"alert": "Person Fall Detected!"}
#                             print(json.dumps(alert_data))  # Print JSON alert
#                             putTextRect(frame, "Fall Detected!", (x1, y1 - 20), scale=2, thickness=3, colorR=color)

#                             # Send email only once
#                             if send_email("hashirkp13@gmail.com", "Fall Detected!", "A person has fallen. Immediate attention needed!"):
#                                 email_sent = True

                      
#                 # Update previous Y position
#                 previous_y_positions[person_id] = y1

#                 # Draw bounding box and label using CVZone
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
#                 putTextRect(frame, label, (x1, y1 - 10), scale=1, thickness=2, colorR=color)

#     # Show video with detections
#     cv2.imshow("Fall Detection System", frame)

#     # Exit on 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()
#---------------------------------
import cv2
import json
from ultralytics import YOLO
from cvzone.Utils import putTextRect
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Email alert function
def send_email(to_email, subject, message):
    sender_email = "support@zoftcares.in"  # Replace with your email
    sender_password = "U_Dr*0R($q&?"        # Replace with your email password

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(message, "plain"))

    try:
        server = smtplib.SMTP_SSL("mail.zoftcares.in", 465)  # Use SSL instead of SMTP
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

# Track previous Y position for fall detection
previous_y_positions = {}
fall_threshold = 130  # Adjust as needed
email_sent = False  # Track whether the email has been sent

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit if no frame

    # Run YOLO inference
    results = model(frame)

    for result in results:
        boxes = result.boxes  # Get detected objects

        for box in boxes:
            class_id = int(box.cls)  # Get class ID
            confidence = float(box.conf)  # Get confidence score
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates

            # Get class name
            class_name = model.names[class_id]
            print(f"Detected: {class_name} (Confidence: {confidence:.2f})")  # Debugging

            if confidence > 0.6:  # Confidence threshold
                person_id = class_id  # Assuming only one person detected at a time

                if class_name == "Walking":
                    color = (0, 255, 0)  # Green for walking
                    label = "Person Walking"
                    email_sent = False  # Reset email flag when person is walking
                elif class_name == "Sitting":
                    color = (255, 255, 0)  # Yellow for sitting
                    label = "Person Sitting"
                elif class_name == "Fall Detected":
                    color = (0, 0, 255)  # Red for fall
                    label = "Person Fallen"
                    print("Fall detected! Checking conditions...")

                    # Track Y position to detect actual fall
                    if person_id in previous_y_positions:
                        y_difference = previous_y_positions[person_id] - y1
                        print(f"Y Difference: {y_difference}")  # Debugging

                        if not email_sent:
                            print("Fall detected! Sending email alert...")
                            alert_data = {"alert": "Person Fall Detected!"}
                            print(json.dumps(alert_data))  # Print JSON alert
                            putTextRect(frame, "Fall Detected!", (x1, y1 - 20), scale=2, thickness=3, colorR=color)

                            # Send email only once per fall event
                            if send_email("hashirkp13@gmail.com", "Fall Detected!", "A person has fallen. Immediate attention needed!"):
                                email_sent = True  

                # Update previous Y position
                previous_y_positions[person_id] = y1

                # Draw bounding box and label using CVZone
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                putTextRect(frame, label, (x1, y1 - 10), scale=1, thickness=2, colorR=color)

    # Show video with detections
    cv2.imshow("Fall Detection System", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

