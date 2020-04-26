import cv2
import numpy as np
import ctypes
import sys
import matplotlib.pyplot as plt
from threading import Thread
import os
import cv2
import torch
from torchvision import models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image


def process_image(im):
    # Scales, crops, and normalizes a PIL image for a PyTorch model,
    # returns an Numpy array
    # Process a PIL image for use in a PyTorch model
    # im = torch.from_numpy(im).long()
    process = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    im = process(im)
    return im


def get_prediction(img, threshold):
    img = process_image(img)
    img = img.unsqueeze(0)
    pred = model(img)  # Pass the image to the model
    print("lakjd")
    scores = pred[0]["scores"].detach().numpy()
    pred_boxes = []
    pred_classes = []
    pred_score = []
    for i in range(len(scores)):
        if scores[i] > threshold:
            pred_score.append(scores[i])
            pred_boxes.append(pred[0]["boxes"].detach().numpy()[i])
            pred_classes.append(
                COCO_INSTANCE_CATEGORY_NAMES[int(pred[0]["labels"].numpy()[i])]
            )
            print("lajsdlkajksld")

    pred_boxes = [
        [(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]["boxes"].detach().numpy())
    ]  # Bounding boxes
    pred_score = list(pred[0]["scores"].detach().numpy())

    pred_t = [
        pred_score.index(x) for x in pred_score if x > threshold
    ]  # Get list of index with score greater than threshold.

    return pred_boxes, pred_classes


ratio = 0.1

model = models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
model.eval()
COCO_INSTANCE_CATEGORY_NAMES = [
    "__background__",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "N/A",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "N/A",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

cap = cv2.VideoCapture("videos/20180206_114604.MP4")

# with open("coco.names", "r") as f:
#     CLASSES = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(COCO_INSTANCE_CATEGORY_NAMES), 3))

while True:
    ret, frame = cap.read()
    image = frame.copy()

    boxes, detections = get_prediction(image, 0.2)
    print("alksjdlakjds")
    for box in boxes:
        # label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)

        startX = int(box[0][0])
        startY = int(box[0][1])
        endX = int(box[1][0])
        endY = int(box[1][1])

        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 3)
        print("alksdlas")
        # y = startY - 15 if startY - 15 > 15 else startY + 15
        # cv2.putText(
        #     frame,
        #     "PERSON" + " - ratio: " + str(round((detection.Area) / (h * w), 2)),
        #     (startX, y),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.5,
        #     COLORS[idx],
        #     2,
        # )
    cv2.namedWindow("Frame", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Frame", frame)
    cv2.resizeWindow("Frame", int(frame.shape[1] / 2), int(frame.shape[0] / 2))
    cv2.waitKey()
