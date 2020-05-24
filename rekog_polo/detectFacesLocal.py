# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# PDX-License-Identifier: MIT-0 (For details, see https://github.com/awsdocs/amazon-rekognition-developer-guide/blob/master/LICENSE-SAMPLECODE.)

import boto3
import json
import cv2
import base64
from searchFaceLocal import search_face_local


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
        crop = np.copy(img)[y:y+h, x:x+h, :]
        # im_arr: image in Numpy one-dim array format.
        _, im_arr = cv2.imencode('.jpg', np.copy(crop))
        im_bytes = im_arr.tobytes()
        crops.append(im_bytes)
        if showCrop:
            cv2.imshow('crop', crop)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return crops


def crop_faces(image, face_locations):
    imwidth = len(image[0])
    imheight = len(image)
    faces = []

    for coords in face_locations:
        leftBorder = int(coords["Left"]*imwidth)
        rightBorder = int(coords["Left"]*imwidth) + \
            int(coords["Width"]*imwidth)
        topBorder = int(coords["Top"]*imheight)
        bottomBorder = int(coords["Top"]*imheight) + \
            int(coords["Height"]*imheight)

        try:

            # cv2.imshow(
            #     "aa", image[topBorder:bottomBorder, leftBorder:rightBorder])
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            faces.append(image[topBorder:bottomBorder, leftBorder:rightBorder])
        except Exception as e:
            print(leftBorder, rightBorder, topBorder, bottomBorder)
            print(e)

    return faces


def main():
    image = cv2.imread("media/todos.jpg")
    imwidth = len(image[0])
    imheight = len(image)
    # cv2.imshow("aa", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    retval, im_bytes = cv2.imencode('.jpg', image)
    im_bytes = base64.b64encode(im_bytes)
    im_bytes = base64.decodebytes(im_bytes)
    face_locations = detect_faces(im_bytes)

    faces = crop_faces(image, face_locations)
    matches = []
    for i in faces:
        try:
            match = search_face_local(i, "myfirstcollection", True)
            matches.append(match)
        except Exception as e:
            print("Exception 1:", e)

    for coords, match in zip(face_locations, matches):
        leftBorder = int(coords["Left"]*imwidth)
        rightBorder = int(coords["Left"]*imwidth) + \
            int(coords["Width"]*imwidth)
        topBorder = int(coords["Top"]*imheight)
        bottomBorder = int(coords["Top"]*imheight) + \
            int(coords["Height"]*imheight)
        cv2.rectangle(image, (leftBorder, topBorder),
                      (rightBorder, bottomBorder), [255, 255, 255], 2)

        try:
            name = match["Face"]["ExternalImageId"]
        except:
            name = ""

        cv2.rectangle(image, (leftBorder, bottomBorder-20),
                      (leftBorder+12*len(name), bottomBorder),
                      [255, 255, 255], cv2.FILLED)
        cv2.putText(image, name.upper(), (leftBorder + 5, bottomBorder-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=1)

    image = cv2.resize(image, (int(imwidth/2), int(imheight/2)))

    cv2.imshow("a", image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
