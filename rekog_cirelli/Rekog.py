import boto3
import cv2
import base64
import time
import json
from boca import passwords
def take_photo():
    cap = cv2.VideoCapture(0)

    retval,img = cap.read()

    _,im_arr = cv2.imencode('.jpg',img)
    im_bytes = im_arr.tobytes()
    cap.release()
    return im_bytes

def compare_faces(key,secret_key,collection_id,im_bytes,maxFaces):

    client = client=boto3.client('rekognition',
    region_name='us-west-2',
    aws_access_key_id=key,
    aws_secret_access_key=secret_key)

    response = client.search_faces_by_image(
        CollectionId = collection_id,
        Image = {
            'Bytes':im_bytes
        },
        MaxFaces = maxFaces
    )

    faceMatches = response['FaceMatches'].Faces
    print(faceMatches)


if __name__ == "__main__":
    key,secret_key = passwords()
    im_bytes = take_photo()
    a = time.time()
    compare_faces(key,secret_key,'reckogcollection',im_bytes,1)
    endtime = time.time() - a
    print(endtime)