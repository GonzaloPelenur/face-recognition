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

print('loading known faces')


print('processing unknown faces')


def start():
    video = cv2.VideoCapture("media/videocolo.mp4")
    frame = 0

    while video.isOpened():
        try:
            rect, image = video.read()
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            locations = face_recognition.face_locations(image, model=MODEL)
            frame += 1
            print(frame)
            for face_location in locations:

                match = None

            #     top_left = (face_location[3]-30, face_location[0]-30)
            #     bottom_right = (face_location[1]+30, face_location[2]+30)

            #     cv2.rectangle(image, top_left, bottom_right,
            #                   (255, 0, 0), FRAME_THICKNESS)

            # cv2.imshow('', image)

            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        except Exception as e:
            print(e)
            break


if __name__ == "__main__":
    a = time.time()
    start()
    print("Detection time:", time.time()-a)
