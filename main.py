import sys

import numpy as np
from cv2 import cv2

import painting_detection
from ccl import draw_components, image_segmentation
from darknet_yolo.darknet_pytorch import Darknet
from htrdc import HTRDC, undistort
from people_detection import DetectNet
from painting_rectification import RectifyClass
from painting_retrieval import RetrieveClass
from utils import resize_when_too_big, is_painting_outside_frame

HTRDC_K_START = 0.0
HTRDC_K_END = 1e-4
HTRDC_N = 20
HTRDC_EPSILON = 1e-6


def compute_HTRDC(img, k):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 15)
    canny = painting_detection.auto_canny(gray)
    print("Computing HTRDC")
    if not k:  # Computes k only for the first frame
        k = HTRDC(canny,
                  (HTRDC_K_START, HTRDC_K_END), HTRDC_N, HTRDC_EPSILON)
    print("Distortion Parameter: ", k)

    return undistort(img, k)


def main():
    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    cap = cv2.VideoCapture("videos/VIRB0392.MP4")
    # cap = cv2.VideoCapture("videos/GOPR5819.MP4")

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    codec = cv2.VideoWriter_fourcc(*"DIVX")
    fps = 30
    out = cv2.VideoWriter("output.avi", 0, fps, (frame_width, frame_height))
    first_frame_flag = True
    k = None

    detect = Darknet()
    retrieve = RetrieveClass()
    rectify = RectifyClass(retrieve)

    # Check if camera opened successfully
    if cap.isOpened() == False:
        print("Error opening video stream or file")

    # Read until video is completed
    while cap.isOpened():

        ret, frame = cap.read()

        if ret == True:
            detection_list = detect.yolo_detection(frame, draw=False)

            for detection in detection_list:
                if detection[0] == 'painting' and not is_painting_outside_frame(
                        frame_width, frame_height, *detection[2:6]):

                    left = detection[2]  # x
                    top = detection[3]  # y
                    right = detection[4]  # x + w
                    bottom = detection[5]  # y + h

                    painting = frame[int(top * 0.75):int(bottom),
                                     int(left):int(right)]
                    cv2.imshow("roi", painting)

                    rectify.rectify(painting)
            #     cv2.imshow("Frame", img2)

            # out.write(img3)
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord("q"):
                break

        # Break the loop
        else:
            break

    # When everything done, release the video capture and writer objects
    cap.release()
    out.release()

    # Closes all the frames
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
