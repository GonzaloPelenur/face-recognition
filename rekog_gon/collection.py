import boto3
import csv

with open('credentials.csv', 'r') as creds:
    next(creds)
    reader = csv.reader(creds)
    for line in reader:
        access_key_id = line[2]
        secret_access_key = line[3]

def create_collection(collection_id):
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

    #Create a collection
    print('Creating collection:' + collection_id)
    response=client.create_collection(CollectionId=collection_id)
    print('Collection ARN: ' + response['CollectionArn'])
    print('Status code: ' + str(response['StatusCode']))
    print('Done...')
    
def main():
    collection_id='Collection'
    create_collection(collection_id)

if __name__ == "__main__":
    main()    