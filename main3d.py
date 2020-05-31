from painting_retrieval import RetrieveClass
from painting_rectification import RectifyClass
import glob
from cv2 import cv2
from darknet_yolo.darknet_pytorch import Darknet
from utils import resize_when_too_big, is_roi_outside_frame




def show_img(img):
    """
    If there is a matched image, the function shows it on screen
        :param img: image to show on screen
    """
    cv2.namedWindow("Img", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Img", img)
    cv2.resizeWindow("Img", int(img.shape[1] / 2), int(img.shape[0] / 2))
    cv2.waitKey(10)


detect = Darknet()


for file in glob.glob("3d_screenshots/*.PNG"):

    frame = cv2.imread(file)
    # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    frame = cv2.flip(frame, 1)
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]
    ######### DETECTION
    detections_list = detect.yolo_detection(frame, False)
    show_img(frame)

    for detection in detections_list:
        if detection[0] == 'painting' and not is_roi_outside_frame(
                frame_width, frame_height, *detection[2:6]):
            print("[INFO] Painting detected: " + file)
            left = detection[2]  # x
            top = detection[3]  # y
            right = detection[4]  # x + w
            bottom = detection[5]  # y + h
            painting = frame[int(top * 0.75):int(bottom), int(left):int(right)]
            show_img(painting)

            retrieve = RetrieveClass()
            rectify = RectifyClass(retrieve)

            coordinates = left,top,right,bottom
            image_3d_warped = rectify.rectify_from_3d(painting,coordinates,frame)
