import face_recognition
import cv2
import requests
import numpy as np
import json
import time

FONT_THINCKNESS= 2
FRAME_THICKNESS =3
MODEL = "cnn"
threshold = 15 #Num of encodings
send_encodigns = []
counter = 0
url = 'http://127.0.0.1:5000/compare'

video = cv2.VideoCapture(0)

while True:
    rect, image = video.read()
    if not rect:
        break
    
    locations = face_recognition.face_locations(image, model=MODEL)
    encodings = face_recognition.face_encodings(image, locations)

    for face_encoding, face_location in zip(encodings, locations):
        if counter <= threshold:
            send_encodigns.append(face_encoding.tolist())
            counter+=1
        else:
            #send the encondings
            data = {'encodings':send_encodigns}
            a = time.time()
            x = requests.post(url,data=json.dumps(data))
            print(x.text)
            print(time.time()-a)
            #reset variables
            send_encodigns = []
            counter = 0
        top_left = (face_location[3], face_location [0])
        bottom_right = (face_location[1], face_location [2])
        color = [0, 0, 0]
        cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)

    cv2.imshow('', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
video.release()