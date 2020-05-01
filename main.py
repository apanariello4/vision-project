import sys

import numpy as np
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') # in order to import cv2 under python3
from cv2 import cv2

import painting_detection
# import painting_retrieval
from htrdc import HTRDC, undistort
# import htrdc

HTRDC_K_START = 0.0
HTRDC_K_END = 1e-4
HTRDC_N = 20
HTRDC_EPSILON = 1e-6


def resize_to_ratio(img, ratio):
    """
    Resize an image according to the given ration
    :param img: Image to be resized
    :param ratio: ratio used to resize the image
    :return: Image resized
    """
    assert ratio > 0, 'ratio_percent must be > 0'
    w = int(img.shape[1] * ratio)
    h = int(img.shape[0] * ratio)
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def resize_when_too_big(img, threshold_w_h):
    h = int(img.shape[0])
    w = int(img.shape[1])
    thr_w, thr_h = threshold_w_h
    if h > thr_h or w > thr_h:
        h_ratio = thr_h / h
        w_ratio = thr_w / w
        ratio = min(h_ratio, w_ratio)
        img = resize_to_ratio(img, ratio)
    return img


def main():
    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    cap = cv2.VideoCapture("videos/VIRB0391.MP4")
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
        ret, frame = cap.read()

        if ret == True:

            # hough_contours = painting_detection.hough_contours(frame.copy())
            # contours, hierarchy = painting_detection.contours(
            #     frame.copy(), adaptive=False)

            # if len(contours) != 0:
            #     painting_detection.draw_contours(
            #         frame.copy(), contours, approximate=False)
            # # img = resize_when_too_big(frame, (720, 405))

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (3, 3), 15)
            canny = painting_detection.auto_canny(gray)
            print("Computing HTRDC")
            if not k:  # Computes k only for the first frame
                k = HTRDC(canny,
                          (HTRDC_K_START, HTRDC_K_END), HTRDC_N, HTRDC_EPSILON)
            print("Distortion Parameter: ", k)

            frame_undistorted = undistort(frame, k)
            cv2.imshow("Frame", frame_undistorted)

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
