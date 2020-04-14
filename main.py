import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') # in order to import cv2 under python3
import cv2
import numpy as np
import painting_detection, painting_retrieval




# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture("videos/VIRB0391.MP4")

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

    if ret == True:

        hough_contours = painting_detection.Hough_Contours(frame.copy())
        contours, hierarchy = painting_detection.Contours(frame.copy(), adaptive=False)

        if len(contours) != 0:
            painting_detection.Draw_Contours(frame, contours, approximate=False)

       
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
