import numpy as np
from datetime import datetime
import face_recognition as fr
from deepface import DeepFace
import cv2
import csv
import pytz
import time
  
now = datetime.now()

current_date = now.strftime("%Y-%m-%d")


video_capture=cv2.VideoCapture(0)

shivaya_image=fr.load_image_file("photos\shivaya.jpeg")
shivaya_encoding=fr.face_encodings(shivaya_image)[0]

sharmila_image=fr.load_image_file("photos\sharmila.jpeg")
sharmila_encoding=fr.face_encodings(sharmila_image)[0]

shanmathi_image=fr.load_image_file("photos\Shanmathi.jpeg")
shanmathi_encoding=fr.face_encodings(shanmathi_image)[0]

dona_image=fr.load_image_file("photos\Dona.jpeg")
dona_encoding=fr.face_encodings(dona_image)[0]

harinee_image=fr.load_image_file("photos\Harinee.jpeg")
harinee_encoding=fr.face_encodings(harinee_image)[0]

known_face_encodings=[shivaya_encoding,sharmila_encoding,shanmathi_encoding,dona_encoding,harinee_encoding]
known_face_names=["Shivaya","Sharmila","Shanmathi","Dona","Harinee"]
face_names = []
students = known_face_names.copy()
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(current_date + '.csv', 'w+',encoding='UTF8', newline='')
lnwriter = csv.writer(f)

while True:
    times=now.strftime('%I:%M:%S')
    ret,frame=video_capture.read()
    rgb_frame=frame[:,:,::-1]
    face_locations=fr.face_locations(rgb_frame)
    face_encodings=fr.face_encodings(rgb_frame,face_locations)
    result = DeepFace.analyze(frame, actions=['emotion'],enforce_detection=False)
    font=cv2.FONT_HERSHEY_SIMPLEX
    for (top, right, bottom, left), face_encoding in zip(face_locations,face_encodings):
        matches=fr.compare_faces(known_face_encodings,face_encoding)

        name="Unknown"
        
        face_distances=fr.face_distance(known_face_encodings,face_encoding)

        best_match_index=np.argmin(face_distances)

        if matches[best_match_index]:
            name=known_face_names[best_match_index]
        if result['dominant_emotion'] in ['happy','surprise','neutral']:
            status='active'
        elif result['dominant_emotion'] in ['fear','disgust','angry','sad']:
            status='Not active'
       
        face_names.append(name)
        if name in known_face_names:
            if name in students:
                students.remove(name)
                lnwriter.writerow([name,times,current_date,status])

        cv2.rectangle(frame,(left,top),(right,bottom),(0,0,255),2)
        
        cv2.rectangle(frame,(left,bottom -35),(right,bottom),(0,0,255),cv2.FILLED)
        
        cv2.putText(frame,name,(left +6,bottom -6),font,1.0,(255,255,255),1)
        

    cv2.imshow('Webcam_facerecognition',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()




         
