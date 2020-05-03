import sys

import numpy as np
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') # in order to import cv2 under python3
from cv2 import cv2

import painting_detection
from ccl import *
from htrdc import HTRDC, undistort
from painting_retrieval import retrieval

# from people_detection import detection

HTRDC_K_START = 0.0
HTRDC_K_END = 1e-4
HTRDC_N = 20
HTRDC_EPSILON = 1e-6


def compute_HTRDC(img):

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
    cap = cv2.VideoCapture("videos/VIRB0416.MP4")
    # cap = cv2.VideoCapture("videos/GOPR5819.MP4")

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    codec = cv2.VideoWriter_fourcc(*"DIVX")
    fps = 30
    out = cv2.VideoWriter("output.avi", 0, fps, (frame_width, frame_height))
    first_frame_flag = True
    k = None

    # Check if camera opened successfully
    if cap.isOpened() == False:
        print("Error opening video stream or file")

    # Read until video is completed
    while cap.isOpened():

        ret, frame = cap.read()

        if ret == True:

            hough_contours = painting_detection.hough_contours(frame.copy())
            contours, hierarchy = painting_detection.contours(
                frame.copy(), adaptive=False)

            if len(contours) != 0:
                painting_detection.draw_contours(
                    frame.copy(), contours, approximate=False)
            # img = resize_when_too_big(frame, (720, 405))
            components = labeling(frame)
            drawn_components = draw_components(components)

            painting_detection.draw_contours(
                drawn_components, contours, approximate=True)

            cv2.imshow("Frame", drawn_components)

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
