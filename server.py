import random
import matplotlib.pyplot as plt
import timeit
import time
import pickle
import cv2
import os
import face_recognition
from flask import Flask, request
from flask_cors import CORS
import json


app = Flask(__name__)
CORS(app)


KNOWN_FACES_DIR = 'known_faces'
TOLERANCE = 0.6
FONT_THINCKNESS = 2
FRAME_THICKNESS = 3
MODEL = "cnn"

video = cv2.VideoCapture(0)
print('loading known faces')

known_faces = []
known_names = []


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


@app.route("/compare", methods=["POST"])
def handle_request():
    results = []
    for i in known_faces:
        recog = face_recognition.compare_faces(
            known_faces, i, TOLERANCE)

        if True in recog:
            match = str(known_names[recog.index(True)])
            results.append(match)
            print(f'Match found: {match}')

    req = json.loads(request.data)
    res = {"status": str(results)}

    return json.dumps(res)


app.run()
