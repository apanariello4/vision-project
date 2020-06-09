import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') # in order to import cv2 under python3
from cv2 import cv2 
import numpy as np
import painting_detection, painting_retrieval
import utils,ccl,ccl2

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture("videos/GOPR2048.MP4")

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
while cap.isOpened():

    ret, frame = cap.read()
    scale_percent = 50  # percent of original size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    if ret == True:

        #hough_contours = painting_detection.hough_contours(frame.copy())
        #contours, hierarchy = painting_detection.contours(frame.copy(), adaptive=False)

        cv2.imshow("Frame", frame)
        cv2.waitKey(0)



        components = ccl.image_segmentation(frame.copy())

        contours, hierarchy = painting_detection.contours(ccl2.labeling(frame), adaptive=True)

        if len(contours) != 0:
            painting_detection.dram_multiple_contours(frame, contours, approximate=False)


        cv2.imshow("Frame", frame)

        # out.write(img3)
        # Press Q on keyboard to  exit
        cv2.waitKey(0)


    # Break the loop
    else:
        break

# When everything done, release the video capture and writer objects
cap.release()
out.release()

# Closes all the frames
cv2.destroyAllWindows()
