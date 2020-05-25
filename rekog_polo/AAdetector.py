import argparse
import torch
from src.config import COCO_CLASSES, colors
import cv2
import numpy as np


# function indexes
# (0, 1): x and y upper left corner position
# (2, 3): x and y lower right corner


def squares_touching(corner, car):
    #print(corner, car)
    touching = False
    if car[0] < corner[2] and car[3] > corner[1]:
        touching = True

    return touching


# RectA.Left < RectB.Right & & RectA.Right > RectB.Left & &
# RectA.Top > RectB.Bottom & & RectA.Bottom < RectB.Top


def get_args():
    parser = argparse.ArgumentParser(
        "EfficientDet: Scalable and Efficient Object Detection implementation by Signatrix GmbH")
    parser.add_argument("--image_size", type=int, default=512,
                        help="The common width and height for all images")
    parser.add_argument("--cls_threshold", type=float, default=0.5)
    parser.add_argument("--nms_threshold", type=float, default=0.5)
    parser.add_argument("--pretrained_model", type=str,
                        default="trained_models/signatrix_efficientdet_coco.pth")
    parser.add_argument("--input", type=str, default="test_videos/input.mp4")
    parser.add_argument("--output", type=str, default="test_videos/output.mp4")

    args = parser.parse_args()
    return args


def detector(image, image_size=512, cls_threshold=0.5, nms_threshold=0.5, pretrained_model='trained_models/signatrix_efficientdet_coco.pth'):
    try:
        model = torch.load(pretrained_model).module
    except:
        model = torch.load(
            pretrained_model, map_location=torch.device('cpu')).module

    if torch.cuda.is_available():
        model.cuda()

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    height, width = image.shape[:2]
    image = image.astype(np.float32) / 255
    image[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
    image[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
    image[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225
    if height > width:
        scale = image_size / height
        resized_height = image_size
        resized_width = int(width * scale)
    else:
        scale = image_size / width
        resized_height = int(height * scale)
        resized_width = image_size

    image = cv2.resize(image, (resized_width, resized_height))

    new_image = np.zeros((image_size, image_size, 3))
    new_image[0:resized_height, 0:resized_width] = image
    new_image = np.transpose(new_image, (2, 0, 1))
    new_image = new_image[None, :, :, :]
    new_image = torch.Tensor(new_image)
    if torch.cuda.is_available():
        new_image = new_image.cuda()
    with torch.no_grad():
        scores, labels, boxes = model(new_image)
        boxes /= scale
    # if boxes.shape[0] == 0:
        # continue
    info = []
    for box_id in range(boxes.shape[0]):
        pred_prob = float(scores[box_id])
        if pred_prob < cls_threshold:
            break
        pred_label = int(labels[box_id])
        xmin, ymin, xmax, ymax = boxes[box_id, :]
        size = (xmax-xmin)*(ymax-ymin)
        size = int(size.item())
        info.append([COCO_CLASSES[pred_label], size, [int(xmin.item()), int(
            ymin.item()), int(xmax.item()), int(ymax.item())], pred_prob])
    return info


if __name__ == "__main__":
    image = cv2.imread('car2.png')
    print(carDetector(image))
