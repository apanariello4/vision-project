import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') # in order to import cv2 under python3
import cv2
import numpy as np

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture('videos/VIRB0391.MP4')

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

codec = cv2.VideoWriter_fourcc(*"DIVX")
fps = 30
out = cv2.VideoWriter('output.avi',0, fps, (frame_width,frame_height))

#exit()
# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
    ret, frame = cap.read()

    if ret == True:

        #edges = cv2.Canny(frame,50,100)
        #lines = cv2.HoughLinesP(edges, 1, np.pi/180, 60, np.array([]), 50, 5)
        # for line in lines:
                #     for x1, y1, x2, y2 in line:
                #         cv2.line(frame, (x1, y1), (x2, y2), (20, 220, 20), 3)
                # #out.write(frame)

        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        _,thresh = cv2.threshold(grayscale,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        #cv2.drawContours(frame, contours, -1, (0,255,0), 3)
        if len(contours) != 0:
            # draw in blue the contours that were founded
            cv2.drawContours(frame, contours, -1, 255, 3)

            # find the biggest countour (c) by the area
            c = max(contours, key = cv2.contourArea)
            x,y,w,h = cv2.boundingRect(c)

            # draw the biggest contour (c) in green
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
       
        cv2.imshow('Frame',frame)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Break the loop
    else: 
        break

# When everything done, release the video capture and writer objects
cap.release()
out.release()

# Closes all the frames
cv2.destroyAllWindows()

