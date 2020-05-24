import face_recognition
import os
import cv2
import pickle
import time
import timeit
import matplotlib.pyplot as plt
import random

KNOWN_FACES_DIR = 'known_faces'
TOLERANCE = 0.6
FONT_THINCKNESS = 2
FRAME_THICKNESS = 3
MODEL = "cnn"

video = cv2.VideoCapture(0)
print('loading known faces')

known_faces = []
known_names = []
colors = []


def encode():
    face_recognition.face_locations(image, model=MODEL)


for name in os.listdir(KNOWN_FACES_DIR):
    print(name)
    for filename in os.listdir(f'{KNOWN_FACES_DIR}\{name}'):
        print(filename)
        encoding = pickle.load(
            open(f'{KNOWN_FACES_DIR}\{name}\{filename}', 'rb'))
        known_faces.append(encoding)
        known_names.append(int(name))

if len(known_names) > 0:
    next_id = max(known_names) + 1
else:
    next_id = 0

print('processing unknown faces')
while True:
    rect, image = video.read()
    a = time.time()
    locations = face_recognition.face_locations(image, model=MODEL)
    b = time.time()
    timeToDetect = b-a

    a = time.time()
    encodings = face_recognition.face_encodings(
        image, locations, num_jitters=1)
    b = time.time()
    timeToEncode = b-a

    for face_encoding, face_location, i in zip(encodings, locations, range(0, len(locations))):
        a = time.time()
        results = face_recognition.compare_faces(
            known_faces, face_encoding, TOLERANCE)
        b = time.time()

        timeToCompare = b-a

        match = None

        if True in results:
            match = str(known_names[results.index(True)])
            print(f'Match found: {match}')
        else:
            match = str(next_id)
            next_id += 1
            known_names.append(match)
            known_faces.append(face_encoding)
            os.mkdir(f'{KNOWN_FACES_DIR}/{match}')
            pickle.dump(face_encoding, open(
                f'{KNOWN_FACES_DIR}/{match}/{match}-{int(time.time())}.pkl', 'wb'))

        top_left = (face_location[3], face_location[0])
        bottom_right = (face_location[1], face_location[2])

        while len(colors) <= i:

            colors.append([random.randint(0, 255), random.randint(
                0, 255), random.randint(0, 255)])

        cv2.rectangle(image, top_left, bottom_right,
                      colors[i], FRAME_THICKNESS)

        top_left = (face_location[3], face_location[2])
        bottom_right = (face_location[1], face_location[2]+22)
        cv2.rectangle(image, top_left, bottom_right, colors[i], cv2.FILLED)
        cv2.putText(image, match, (face_location[3]+10, face_location[2]+15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

        print("Time to:\ndetect=", timeToDetect, "\nencode=",
              timeToEncode, "\ncompare=", timeToCompare)

    cv2.imshow('', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

plt.show()
cv2.destroyAllWindows()
video.release()
