import face_recognition
import os
import cv2
import pickle
import time
import json

KNOWN_FACES_DIR = 'known_faces'
regiter_name = 'gon'
TOLERANCE = 0.5
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

try:
    with open('known_names.txt') as json_file:
        known_names = json.load(json_file)
except:
    print('except')
    known_names = {}

if len(known_id) > 0:
    next_id = max(known_id) + 1
else:
    next_id = 0

print('processing unknown faces')
while True:
    rect, image = video.read()
    if not rect:
        break
    
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
            known_names.setdefault(match,regiter_name)
            known_faces.append(face_encoding)
            os.mkdir(f'{KNOWN_FACES_DIR}/{match}')
            pickle.dump(face_encoding, open(f'{KNOWN_FACES_DIR}/{match}/{match}-{int(time.time())}.pkl', 'wb'))
            print(f'id: {match} added to {regiter_name}')
        
        top_left = (face_location[3], face_location [0])
        bottom_right = (face_location[1], face_location [2])
        color = [0, 0, 0]
        cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)

        top_left = (face_location[3], face_location [2])
        bottom_right = (face_location[1], face_location [2]+22)
        cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
        cv2.putText(image, match, (face_location[3]+10, face_location[2]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))

    cv2.imshow('', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
video.release()
print(known_names)
print('saving new faces')
with open('known_names.txt', 'w') as outfile:
    json.dump(known_names, outfile)