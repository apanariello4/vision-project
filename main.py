import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') # in order to import cv2 under python3
import cv2
import numpy as np

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture('videos/VIRB0392.MP4')

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

codec = cv2.VideoWriter_fourcc(*"DIVX")
fps = 30
out = cv2.VideoWriter('output.avi',0, fps, (frame_width,frame_height))
og = True

# Check if camera opened successfully
if (cap.isOpened()== False): 
    print("Error opening video stream or file")

# Read until video is completed
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    prev_img = None

    if ret == True:

        #edges = cv2.Canny(frame,50,100)
        #lines = cv2.HoughLinesP(edges, 1, np.pi/180, 60, np.array([]), 50, 5)
        # for line in lines:
            #     for x1, y1, x2, y2 in line:
            #         cv2.line(frame, (x1, y1), (x2, y2), (20, 220, 20), 3)

        #---------------Saliency---------------------------------------
        # initialize OpenCV's static fine grained saliency detector and
        # compute the saliency map
        saliency = cv2.saliency.StaticSaliencyFineGrained_create()
        (success, saliencyMap) = saliency.computeSaliency(frame)
        # if we would like a *binary* map that we could process for contours,
        # compute convex hull's, extract bounding boxes, etc., we can
        # additionally threshold the saliency map

        threshMap = cv2.threshold(saliencyMap.astype("uint8"), 0, 255, cv2.THRESH_BINARY)[1]
        #----------------------------------------------------------------

        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
        _,thresh = cv2.threshold(grayscale[:,:,2],10,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25,25))
        #opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)

        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) != 0:
            # draw in blue the contours that were founded
            cv2.drawContours(frame, contours, -1, 255, 3)

            # find the biggest countour (c) by the area
            c = max(contours, key = cv2.contourArea)

            # approximate the contour (approx) with 10% tolerance
            epsilon = 0.1*cv2.arcLength(c,True)
            approx = cv2.approxPolyDP(c,epsilon,True)
            x,y,w,h = cv2.boundingRect(approx)
            
            # draw the biggest contour (c) in green
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

            # draw the minimum rectangle around the biggest contour (c) by modifying rotation
            rect = cv2.minAreaRect(approx)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(frame,[box],0,(0,0,255),2)

            if og:
                og_points = np.float32([[x, y],[x+w, y+h],[x, y+h],[x+w, y]])
                og = False
        
        current_bb = np.float32([[x, y],[x+w, y+h],[x, y+h],[x+w, y]])

        M = cv2.getPerspectiveTransform(og_points, current_bb)
        imageWarped = cv2.warpPerspective(frame, M, (frame_width, frame_height))
        #out.write(imageWarped)
        cv2.imshow('Frame',frame)
        cv2.imshow('Warped',imageWarped)

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

