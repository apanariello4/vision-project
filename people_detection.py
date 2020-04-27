import numpy as np
import cv2
import torch.hub
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
import time


def show(img):
    """
        :param img: image to show on screen
    """
    if img.size != 0:
        cv2.namedWindow('People Detection', cv2.WINDOW_KEEPRATIO)
        cv2.imshow('People Detection', img)
        cv2.resizeWindow('People Detection', int(img.shape[1] / 2), int(img.shape[0] / 2))
    else:
        print("No match...")


class DetectNet:
    def __init__(self, threshold=0.5):
        self.classes = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
            'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        self.model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        self.threshold = threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def detect(self, img):
        transform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.ToTensor()], )
        img_t = transform(img).to(self.device)

        with torch.no_grad():
            pred = self.model([img_t])

        pred_class = [self.classes[i] for i in list(pred[0]['labels'].cpu().clone().numpy())]
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().clone().numpy())]
        pred_score = list(pred[0]['scores'].detach().cpu().clone().numpy())
        pred_t = [pred_score.index(x) for x in pred_score if x > self.threshold][-1]
        pred_boxes = pred_boxes[:pred_t + 1]
        pred_class = pred_class[:pred_t + 1]

        for i in range(len(pred_boxes)):
            if pred_class[i] == 'person':
                left = int(pred_boxes[i][0][0])
                top = int(pred_boxes[i][0][1])
                right = int(pred_boxes[i][1][0])
                bottom = int(pred_boxes[i][1][1])
                # color = self.colors[pred_colors[i] % len(self.colors)]
                self.draw_boxes(img, pred_class[i], pred_score[i], left, top, right, bottom, (0, 0, 255))

    def draw_boxes(self, img, class_id, score, left, top, right, bottom, color):
        txt_color = (0, 0, 0)
        if sum(color) < 500:
            txt_color = (255, 255, 255)

        cv2.rectangle(img, (left, top), (right, bottom), color=color, thickness=3)

        label = '{}%'.format(round((score * 100), 1))
        if self.classes:
            label = '%s %s' % (class_id, label)

        label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, label_size[1])
        cv2.rectangle(img, (left, top - round(1.5 * label_size[1])),
                      (left + round(1.5 * label_size[0]), top + base_line), color=color, thickness=cv2.FILLED)
        cv2.putText(img, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color=txt_color, thickness=2)

    def process_image(self, img):
        # Scales, crops, and normalizes a PIL image for a PyTorch model,
        # returns an Numpy array
        # Process a PIL image for use in a PyTorch model
        # im = torch.from_numpy(im).long()
        process = transforms.Compose(
            [
                transforms.ToPILImage(),
                # transforms.Resize(256),
                # transforms.CenterCrop(224),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        img = process(img)
        return img


net = DetectNet()
cap = cv2.VideoCapture("videos/20180206_114604.MP4")

while True:
    ret, frame = cap.read()
    image = frame.copy()

    start = time.time()
    net.detect(image)
    end = time.time()

    cv2.putText(image, '{:.2f}ms'.format((end - start) * 1000), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0),
                2)

    show(image)
    # cv2.imshow('People Detection', frame)
    print("FPS {:5.2f}".format(1 / (end - start)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
