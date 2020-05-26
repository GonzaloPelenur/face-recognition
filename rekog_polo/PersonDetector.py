from AAdetector import detector
import cv2

frameNum = 0
cap = cv2.VideoCapture(0)
while cap.isOpened():
    flag, image = cap.read()
    frameNum += 1

    if frameNum % 15 is 0:
        if not flag:
            break

        info = detector(image)

        for detections in info:
            if (detections[0] is 'person') and detections[1] > 100000:
                print(detections[2])
                cv2.rectangle(image, (detections[2][0], detections[2][1]), (
                    detections[2][2], detections[2][3]), (0, 0, 0), 2)

    cv2.imshow('frame', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
