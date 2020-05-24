import boto3
import csv
import time

BUCKET = "gon-s3"
KEY = "gon.jpg"
COLLECTION = "Collection"

photo = 'elon&gon2.jpg'
with open(photo, 'rb') as source_image:
    source_bytes = source_image.read()

with open('credentials.csv', 'r') as creds:
    next(creds)
    reader = csv.reader(creds)
    for line in reader:
        access_key_id = line[2]
        secret_access_key = line[3]

def search_faces_by_image(bucket, key, collection_id, threshold=80, region="eu-west-1"):
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
	return response['FaceMatches']

a = time.time()
for record in search_faces_by_image(BUCKET, KEY, COLLECTION):
    face = record['Face']
    print(f"Matched Face {record['Similarity']}")
    print(f"  FaceId : {face['FaceId']}")
    print(f"  ImageId : {face['ExternalImageId']}")
    print(f"  BoundingBox : {face['BoundingBox']}")

print(time.time() - a)