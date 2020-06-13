from painting_retrieval import RetrieveClass
from painting_rectification import RectifyClass
import glob
from cv2 import cv2
from darknet_yolo.darknet_pytorch import Darknet
from utils import is_roi_outside_frame, show_img

detect = Darknet()

for file in glob.glob("resources/3d_screenshots/*"):

    frame = cv2.imread(file)
    # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    # frame = cv2.flip(frame, 1)
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]
    frame_copy = frame.copy()
    ######### DETECTION
    detections_list = detect.yolo_detection(frame_copy)
    show_img("3D model image", frame_copy)

    for detection in detections_list:
        if detection[0] == 'painting' and not is_roi_outside_frame(
                frame_width, frame_height, *detection[2:6]):
            print("[INFO] Painting detected: " + file)
            left = detection[2]  # x
            top = detection[3]  # y
            right = detection[4]  # x + w
            bottom = detection[5]  # y + h
            painting = frame[int(top * 0.75):int(bottom), int(left):int(right)]
            show_img("3D model image", painting)

            retrieve = RetrieveClass()
            rectify = RectifyClass(retrieve)

            coordinates = left, top, right, bottom
            image_3d_warped = rectify.rectify_from_3d(painting, coordinates, frame)
