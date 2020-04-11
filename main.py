import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture('videos/VIRB0391.MP4')

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


        img1 = frame
        #img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img1 = cv2.resize(img1, (960, 540))
        img2 = cv2.imread("foto1.png", cv2.IMREAD_GRAYSCALE)


        sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.01, edgeThreshold=20)

        kp1, dest1 = sift.detectAndCompute(img1, None)
        kp2, dest2 = sift.detectAndCompute(img2, None)

        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(dest1, dest2, k=2)

        # Apply ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.3 * n.distance:
                good.append([m])

        # cv.drawMatchesKnn expects list of lists as matches.
        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)



        #        plt.imshow(frame)
        cv2.imshow('Frame',img3)

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

