import boto3
import csv

def add_faces_to_collection(bucket,photo,collection_id):
    with open('credentials.csv', 'r') as creds:
        next(creds)
        reader = csv.reader(creds)
        for line in reader:
            access_key_id = line[2]
            secret_access_key = line[3]
    client = boto3.client('rekognition',
                        region_name='us-east-1',
                        aws_access_key_id=access_key_id,
                        aws_secret_access_key=secret_access_key)

    response=client.index_faces(CollectionId=collection_id,
                                Image={'S3Object':{'Bucket':bucket,'Name':photo}},
                                ExternalImageId=photo,
                                MaxFaces=1,
                                QualityFilter="AUTO",
                                DetectionAttributes=['ALL'])

    print ('Results for ' + photo) 	
    print('Faces indexed:')						
    for faceRecord in response['FaceRecords']:
         print('  Face ID: ' + faceRecord['Face']['FaceId'])
         print('  Location: {}'.format(faceRecord['Face']['BoundingBox']))

    print('Faces not indexed:')
    for unindexedFace in response['UnindexedFaces']:
        print(' Location: {}'.format(unindexedFace['FaceDetail']['BoundingBox']))
        print(' Reasons:')
        for reason in unindexedFace['Reasons']:
            print('   ' + reason)
    return len(response['FaceRecords'])

def main():
    bucket='gon-s3'
    collection_id='Collection'
    photo='elon.jpg'
    
    
    indexed_faces_count=add_faces_to_collection(bucket, photo, collection_id)
    print("Faces indexed count: " + str(indexed_faces_count))


if __name__ == "__main__":
    main()