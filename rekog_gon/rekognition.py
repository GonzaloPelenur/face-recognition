import csv
import boto3
import cv2
import base64
import time

'''
cap = cv2.VideoCapture(0)
retval, img = cap.read()
_, im_arr = cv2.imencode('.jpg', img)  # im_arr: image in Numpy one-dim array format.
im_bytes = im_arr.tobytes()
cap.release()
'''

with open('credentials.csv', 'r') as creds:
    next(creds)
    reader = csv.reader(creds)
    for line in reader:
        access_key_id = line[2]
        secret_access_key = line[3]


photo = 'elon3.jpg'
with open(photo, 'rb') as source_image:
    source_bytes = source_image.read()

client = boto3.client('rekognition',
                        region_name='us-east-1',
                        aws_access_key_id=access_key_id,
                        aws_secret_access_key=secret_access_key)


a = time.time()
response = client.compare_faces(
    SourceImage={
        'S3Object': {
            'Bucket': 'gon-s3',
            'Name': 'elon.jpg'
        }
    },
    TargetImage={
        #'Bytes': im_bytes
        'Bytes': source_bytes
    }
)
print('Response time:',time.time() - a)
print(response['FaceMatches'])

'''
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

