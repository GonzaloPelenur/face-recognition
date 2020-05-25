# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# PDX-License-Identifier: MIT-0 (For details, see https://github.com/awsdocs/amazon-rekognition-developer-guide/blob/master/LICENSE-SAMPLECODE.)

import boto3
import cv2
import base64


# if convert is True it converts image into bytes
def search_face_local(image, collectionId, matches, location, convert=False):
    if(convert):
        retval, image = cv2.imencode('.jpg', image)
        image = base64.b64encode(image)
        image = base64.decodebytes(image)

    threshold = 70
    maxFaces = 100

    client = boto3.client('rekognition')

    try:
        response = client.search_faces_by_image(CollectionId=collectionId,
                                                Image={'Bytes': image},
                                                FaceMatchThreshold=threshold,
                                                MaxFaces=maxFaces,
                                                )
        faceMatches = response['FaceMatches']
        matches.append([faceMatches[0], location, image])
    except Exception as e:
        matches.append(["unknown", location, image])
        return []

    # print(faceMatches[0]["Face"]["ExternalImageId"])
    # print('Similarity: ' + "{:.2f}".format(faceMatches[0]['Similarity']) + "%")

    # for match in faceMatches:
    #     print('FaceId:' + match['Face']["ExternalImageId"])
    #     print('Similarity: ' + "{:.2f}".format(match['Similarity']) + "%")
    #     print
    return faceMatches[0]


if __name__ == "__main__":
    image = cv2.imread("media/mai.jpg")
    # cv2.imshow("cena", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    retval, im_bytes = cv2.imencode('.jpg', image)
    im_bytes = base64.b64encode(im_bytes)
    im_bytes = base64.decodebytes(im_bytes)

    collectionId = 'myfirstcollection'
    facemathces = search_face_local(im_bytes, collectionId)
