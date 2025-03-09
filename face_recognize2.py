#----------ith face recognize maathram aan
import cv2
import face_recognition

known_face_encoding=[]
known_face_names=[]

#load images
known_person1_img=face_recognition.load_image_file('img10.jpg') # hafeefa image
known_person2_img=face_recognition.load_image_file('img18.jpg') # shameer image

# that image will encode 
known_person1_encoding=face_recognition.face_encodings(known_person1_img)[0]
known_person2_encoding=face_recognition.face_encodings(known_person2_img)[0]

known_face_encoding.append(known_person1_encoding)
known_face_encoding.append(known_person2_encoding)
# print(known_face_encoding)

known_face_names.append('hafeefa') # append 1 argument olu idka 2 enam ndekil [] ith idnm
known_face_names.append('shameer')
#initialize the webcam
video_capture=cv2.VideoCapture(0)
if not video_capture.isOpened():
    print("error:video not opened")
    exit()

while True:
    ret,frame=video_capture.read()
    rgb_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    face_locations=face_recognition.face_locations(rgb_frame)
    face_encoding=face_recognition.face_encodings(rgb_frame,face_locations)
    # print(face_encoding)
        ##The zip() function pairs each face location tuple (right, left, top, bottom) with its corresponding face encoding.
          # This ensures that each detected face is processed with its corresponding encoding.
          # Extracts the corresponding encoding for that face
    for (right,left,top,bottom),face_encode in zip(face_locations,face_encoding):
        matches=face_recognition.compare_faces(known_face_encoding,face_encode)
        # print(matches)
        name="Unknown"

        if True in matches:
            first_match_index=matches.index(True)
            name=known_face_names[first_match_index]
        cv2.rectangle(frame,(left,top),(right,bottom),(0,0,255),2)
        cv2.putText(frame,name,(left,top-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,255),2)

    # display the resulting frame
    cv2.imshow("video",frame)

    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()