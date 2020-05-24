import boto3
import csv
import time
import cv2
import numpy as np

def search_faces_by_image(access_key_id, secret_access_key, source_bytes, collection_id, threshold=80, region="eu-west-1"):
    rekognition = boto3.client('rekognition',
                        region_name='us-east-1',
                        aws_access_key_id=access_key_id,
                        aws_secret_access_key=secret_access_key)
    response = rekognition.search_faces_by_image(
		Image={
            'Bytes': source_bytes
		},
		CollectionId=collection_id,
		FaceMatchThreshold=threshold,
        MaxFaces=1
	)

    for record in response['FaceMatches']:
        face = record['Face']
        print(f"Matched Face {record['Similarity']}")
        print(f"  FaceId : {face['FaceId']}")
        print(f"  ImageId : {face['ExternalImageId']}")
        print(f"  BoundingBox : {face['BoundingBox']}")
    return response['FaceMatches']

def detect_faces(im_path):
    crops = []
    face_cascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
    img = cv2.imread(im_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        #aca armar los crops y apendearlos
        crop = np.copy(img)[y:y+h,x:x+h,:]
        cv2.imshow('img', crop)
        cv2.waitKey()
        cv2.destroyAllWindows()
        
        _, im_arr = cv2.imencode('.jpg', np.copy(crop))  # im_arr: image in Numpy one-dim array format.
        im_bytes = im_arr.tobytes()
        crops.append(im_bytes)

    return crops

def main():
    COLLECTION = "Collection"
    im_path = 'media/elon&gon2.jpg'

    with open('credentials.csv', 'r') as creds:
        next(creds)
        reader = csv.reader(creds)
        for line in reader:
            access_key_id = line[2]
            secret_access_key = line[3]

    a = time.time()
    for source_bytes in detect_faces(im_path):
        #thread
        search_faces_by_image(access_key_id, secret_access_key, source_bytes, COLLECTION)
    print(time.time() - a)

if __name__ == "__main__":
    main()
    #detect_faces('media/elon&gon2.jpg')