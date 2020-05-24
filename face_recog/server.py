import random
import timeit
import time
import pickle
import cv2
import os
import face_recognition
from flask import Flask, request
from flask_cors import CORS
import json
import numpy as np


app = Flask(__name__)
CORS(app)


KNOWN_FACES_DIR = 'known_faces'
TOLERANCE = 0.5
FONT_THINCKNESS = 2
FRAME_THICKNESS = 3
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


@app.route("/compare", methods=["POST"])
def handle_request():
    req = json.loads(request.data)
    encodings = req['encodings']
    response = []
    for i in encodings:
        face_encoding = np.array(i)
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

        if len(known_names) > int(match):
            response.append(known_names[match]) 
        else:
            response.append('Unknown')
    
    res = {"status": response}
    print(res)

    return json.dumps(res)
app.run()
