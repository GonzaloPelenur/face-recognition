# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# PDX-License-Identifier: MIT-0 (For details, see https://github.com/awsdocs/amazon-rekognition-developer-guide/blob/master/LICENSE-SAMPLECODE.)

import boto3
import json
import cv2
import base64
from searchFaceLocal import search_face_local
import time
import face_recognition
import numpy as np
import threading


def detect_faces(im_bytes):

    client = boto3.client('rekognition')
    face_locations = []
    response = client.detect_faces(
        Image={'Bytes': im_bytes}, Attributes=['DEFAULT'])

    print('Detected faces for images')
    for faceDetail in response['FaceDetails']:
        face_locations.append(faceDetail["BoundingBox"])

    return face_locations


def detect_faces_cascade(image, showCrop=False):
    crops = []
    face_cascade = cv2.CascadeClassifier(
        'Cascades/haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        # aca armar los crops y apendearlos
        crop = np.copy(image)[y:y+h, x:x+h, :]
        # im_arr: image in Numpy one-dim array format.
        _, im_arr = cv2.imencode('.jpg', np.copy(crop))
        im_bytes = im_arr.tobytes()
        crops.append(im_bytes)
        if showCrop:
            cv2.imshow('crop', crop)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return crops


def detect_faces_recog(image, showCrop=False):
    crops = []

    locations = face_recognition.face_locations(image, model="cnn")
    for (top, right, bottom, left) in locations:
        # aca armar los crops y apendearlos
        crop = image[top-10:bottom+10, left-10:right+10]
        crops.append(crop)
        if showCrop:
            cv2.imshow('crop', crop)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return crops, locations


def display(image, matches):
    for match in matches:
        location = match[1]
        try:
            name = match[0]["Face"]["ExternalImageId"]
        except:
            name = ""

        top_left = (location[3], location[0])
        bottom_right = (location[1], location[2])
        color = [0, 0, 0]
        cv2.rectangle(image, top_left, bottom_right, color, 2)

        top_left = (location[3], location[2])
        bottom_right = (location[1], location[2]+22)
        cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
        cv2.putText(image, name, (location[3]+5, location[2]+15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

    cv2.imshow("recognition", image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    threads = []

    image = cv2.imread("media/idtc2.jpg")

    crops, locations = detect_faces_recog(image, False)

    matches = []
    # NOTE: without threading
    # for i in crops:
    #     match = search_face_local(i, "myfirstcollection", matches, True)
    #     matches.append(match)

    # NOTE: with threading
    for crop, location in zip(crops, locations):
        thread = threading.Thread(
            target=search_face_local, args=(crop, "myfirstcollection", matches, location, True))
        thread.start()
        threads.append(thread)

    for process in threads:
        process.join()

    display(image, matches)


if __name__ == "__main__":
    a = time.time()
    main()
    print(time.time()-a)
