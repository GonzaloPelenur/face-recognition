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
    print(encoding)
    for i in encodings:
        face_encodings = np.array(i)
        recog = face_recognition.compare_faces(known_faces, face_encodings, TOLERANCE)
        if True in recog:
            match = str(known_id[recog.index(True)])
            response.append(match)
            print(f'Match found: {match}')

    
    res = {"status": str(response)}

    return json.dumps(res)
app.run()
