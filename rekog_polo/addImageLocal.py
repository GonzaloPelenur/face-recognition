# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# PDX-License-Identifier: MIT-0 (For details, see https://github.com/awsdocs/amazon-rekognition-developer-guide/blob/master/LICENSE-SAMPLECODE.)

import boto3
import cv2
import base64


def add_faces_to_collection(image, collection_id, label, convert=False):
    if(convert):
        retval, image = cv2.imencode('.jpg', image)
        image = base64.b64encode(image)
        image = base64.decodebytes(image)

    client = boto3.client('rekognition')

    response = client.index_faces(CollectionId=collection_id,
                                  Image={'Bytes': image},
                                  ExternalImageId=label,
                                  MaxFaces=3,
                                  QualityFilter="AUTO",
                                  DetectionAttributes=['ALL'])

    print('Results for ' + label)
    print('Faces indexed:')
    for faceRecord in response['FaceRecords']:
        print('  Face ID: ' + faceRecord['Face']['ExternalImageId'])
        print('  Location: {}'.format(faceRecord['Face']['BoundingBox']))

    print('Faces not indexed:')
    for unindexedFace in response['UnindexedFaces']:
        print(' Location: {}'.format(
            unindexedFace['FaceDetail']['BoundingBox']))
        print(' Reasons:')
        for reason in unindexedFace['Reasons']:
            print('   ' + reason)
    return len(response['FaceRecords'])


def main():
    image = cv2.imread("media/colo.jpg")
    # cv2.imshow("cena", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    label = "maxi_sucari"
    collection_id = "myfirstcollection"

    indexed_faces_count = add_faces_to_collection(
        image, collection_id, label, True)

    print("Faces indexed count: " + str(indexed_faces_count))


if __name__ == "__main__":
    main()
