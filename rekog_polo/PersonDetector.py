from AAdetector import detector
import cv2
import time


def start(video=0, frameRate=15):
    frameNum = 0
    cap = cv2.VideoCapture(video)

    while cap.isOpened():

        flag, image = cap.read()
        frameNum += 1

        if frameNum % frameRate is 0:
            if not flag:
                break

            info = detector(image)

            for detections in info:
                if (detections[0] is 'person') and detections[1] > 100000:
                    continue
        #             print(detections[2])
        #             cv2.rectangle(image, (detections[2][0], detections[2][1]), (
        #                 detections[2][2], detections[2][3]), (0, 0, 0), 2)

        # cv2.imshow('frame', image)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break


if __name__ == "__main__":
    frate = 1
    a = time.time()
    start("media/videocolo.mp4", frate)
    print("Detection framerate =", frate, "Time:", time.time()-a)

    # frate = 5
    # a = time.time()
    # start("media/videocolo.mp4", frate)
    # print("Detection framerate =", frate, "Time:", time.time()-a)

    # frate = 3
    # a = time.time()
    # start("media/videocolo.mp4", frate)
    # print("Detection framerate =", frate, "Time:", time.time()-a)

    # frate = 1
    # a = time.time()
    # start("media/videocolo.mp4", frate)
    # print("Detection framerate =", frate, "Time:", time.time()-a)
