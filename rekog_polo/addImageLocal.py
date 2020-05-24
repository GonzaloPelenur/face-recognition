# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# PDX-License-Identifier: MIT-0 (For details, see https://github.com/awsdocs/amazon-rekognition-developer-guide/blob/master/LICENSE-SAMPLECODE.)

import boto3
import cv2
import base64


def add_faces_to_collection(im_bytes, collection_id, label):

    client = boto3.client('rekognition')

    response = client.index_faces(CollectionId=collection_id,
                                  Image={'Bytes': im_bytes},
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
    image = cv2.imread("media/todos.jpg")
    # cv2.imshow("cena", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    retval, im_bytes = cv2.imencode('.jpg', image)
    im_bytes = base64.b64encode(im_bytes)
    im_bytes = base64.decodebytes(im_bytes)
    label = "maxi_sucari"
    collection_id = "myfirstcollection"
    indexed_faces_count = add_faces_to_collection(
        im_bytes, collection_id, label)
    print("Faces indexed count: " + str(indexed_faces_count))


if __name__ == "__main__":
    main()
