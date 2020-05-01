import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') # in order to import cv2 under python3
from cv2 import cv2
import numpy as np
import painting_detection, painting_retrieval, ccl
import glob
import matplotlib.pyplot as plt

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
videolist = glob.glob('videos/2*.mp4')

for video in videolist:
    cap = cv2.VideoCapture(video)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    codec = cv2.VideoWriter_fourcc(*"DIVX")
    fps = 30
    out = cv2.VideoWriter("output.avi", 0, fps, (frame_width, frame_height))
    first_frame_flag = True


    # Check if camera opened successfully
    if cap.isOpened() == False:
        print("Error opening video stream or file")

    # Read until video is completed


    time=100
    while cap.isOpened():
        cap.set(1,time)
        ret, frame = cap.read()
        time += 10
        frame = cv2.resize(frame, (960, 540))

        if ret == True:

            ccl_labeled = ccl.labeling(frame.copy())
            #cv2.imshow('labeled.png', frame)
            #cv2.waitKey()
            hough_contours = painting_detection.hough_contours(ccl_labeled, frame.copy())
            contours, hierarchy = painting_detection.contours(ccl_labeled, adaptive=True)
            print(contours)
            if len(contours) != 0:

                #painting_detection.draw_contours(frame, contours, approximate=False)
                painting_detection.draw_contours(frame, contours, approximate=False)
            cv2.imshow("Frame", frame)

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
