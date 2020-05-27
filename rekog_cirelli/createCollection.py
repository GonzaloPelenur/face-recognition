import boto3
from boca import passwords

def create_collection(collection_id):

    key,secret_key = passwords()


    client=boto3.client('rekognition',
    region_name='us-west-2',
    aws_access_key_id=key,
    aws_secret_access_key=secret_key)

    #Create a collection
    print('Creating collection:' + collection_id)
    response=client.create_collection(CollectionId=collection_id)
    print('Collection ARN: ' + response['CollectionArn'])
    print('Status code: ' + str(response['StatusCode']))
    print('Done...')
    
def main():
    collection_id='reckogcollection'
    create_collection(collection_id)

if __name__ == "__main__":
    main()