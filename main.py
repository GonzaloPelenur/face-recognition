import face_recognition
import os
import cv2
import pickle
import time
import json

KNOWN_FACES_DIR = 'known_faces'
TOLERANCE = 0.6
FONT_THINCKNESS= 2
FRAME_THICKNESS =3
MODEL = "cnn"

video = cv2.VideoCapture(0)

print('loading known faces')

known_faces = []
known_id = []

for name in os.listdir(KNOWN_FACES_DIR):
    for filename in os.listdir(f'{KNOWN_FACES_DIR}\{name}'):
        encoding = pickle.load(open(f'{KNOWN_FACES_DIR}\{name}\{filename}', 'rb'))
        known_faces.append(encoding)
        known_id.append(int(name))

if len(known_id) > 0:
    next_id = max(known_id) + 1
else:
    next_id = 0

with open('known_names.txt') as json_file:
    known_names = json.load(json_file)

print('processing unknown faces')
while True:
    rect, image = video.read()
    
    locations = face_recognition.face_locations(image, model=MODEL)
    encodings = face_recognition.face_encodings(image, locations)

    for face_encoding, face_location in zip(encodings, locations):
        results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
        match = None

        if True in results:
            match = str(known_id[results.index(True)])
            #print(f'Match found: {match}')
        else:
            match = str(next_id)
            next_id += 1
            known_id.append(match)
            known_faces.append(face_encoding)
            #os.mkdir(f'{KNOWN_FACES_DIR}/{match}')
            #pickle.dump(face_encoding, open(f'{KNOWN_FACES_DIR}/{match}/{match}-{int(time.time())}.pkl', 'wb'))
        
        top_left = (face_location[3], face_location [0])
        bottom_right = (face_location[1], face_location [2])
        color = [0, 0, 0]
        cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)

        top_left = (face_location[3], face_location [2])
        bottom_right = (face_location[1], face_location [2]+22)
        if len(known_names) > int(match):
            name = known_names[match]
        else:
            name = 'Unknown'
            print(match)
        cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
        cv2.putText(image, name, (face_location[3]+10, face_location[2]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))

    cv2.imshow('', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
video.release()