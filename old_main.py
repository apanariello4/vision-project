import sys

sys.path.remove(
    "/opt/ros/kinetic/lib/python2.7/dist-packages"
)  # in order to import cv2 under python3
import cv2
import numpy as np

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture("videos/VIRB0391.MP4")

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

codec = cv2.VideoWriter_fourcc(*"DIVX")
fps = 30
out = cv2.VideoWriter("output.avi", 0, fps, (frame_width, frame_height))
first_frame_flag = True


def Contours(frame):

    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gaussian = cv2.GaussianBlur(grayscale, (3, 3), 100)
    _, thresh = cv2.threshold(
        gaussian, 10, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25,25))
    # opening = cv2.morpholfirst_frameyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)

    return cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


# Check if camera opened successfully
if cap.isOpened() == False:
    print("Error opening video stream or file")

# Read until video is completed
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret == True:

        # edges = cv2.Canny(frame,50,100)
        # lines = cv2.HoughLinesP(edges, 1, np.pi/180, 60, np.array([]), 50, 5)
        # for line in lines:
        #     for x1, y1, x2, y2 in line:
        #         cv2.line(frame, (x1, y1), (x2, y2), (20, 220, 20), 3)

        # ---------------Saliency---------------------------------------
        # initialize OpenCV's static fine grained saliency detector and
        # compute the saliency map
        # saliency = cv2.saliency.StaticSaliencyFineGrained_create()
        # (success, saliencyMap) = saliency.computeSaliency(frame)
        # if we would like a *binary* map that we could process for contours,
        # compute convex hull's, extract bounding boxes, etc., we can
        # additionally threshold the saliency map

        # threshMap = cv2.threshold(saliencyMap.astype("uint8"), 0, 255, cv2.THRESH_BINARY)[1]
        # ----------------------------------------------------------------

        contours, hierarchy = Contours(frame)

        if len(contours) != 0:
            # draw in blue the contours that were founded
            cv2.drawContours(frame, contours, -1, 255, 3)

            # find the biggest countour (c) by the area
            c = max(contours, key=cv2.contourArea)

            # determine the most extreme points along the contour
            extLeft = tuple(c[c[:, :, 0].argmin()][0])
            extRight = tuple(c[c[:, :, 0].argmax()][0])
            extTop = tuple(c[c[:, :, 1].argmin()][0])
            extBot = tuple(c[c[:, :, 1].argmax()][0])

            # cv2.circle(frame, extLeft, 8, (0, 0, 255), -1)
            # cv2.circle(frame, extRight, 8, (0, 255, 0), -1)
            # cv2.circle(frame, extTop, 8, (255, 0, 0), -1)
            # cv2.circle(frame, extBot, 8, (255, 255, 0), -1)

            # approximate the contour (approx) with 10% tolerance
            epsilon = 0.05 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, True)
            x, y, w, h = cv2.boundingRect(approx)

            n = approx.ravel()
            i = 0
            font = cv2.FONT_HERSHEY_COMPLEX
            current_bb = []
            for j in n:
                if i % 2 == 0:
                    _x = n[i]
                    _y = n[i + 1]

                    current_bb.append([_x, _y])
                i += 1

            # draw the biggest contour (c) in green
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # draw the minimum rectangle around the biggest contour (c) by modifying rotation
            rect = cv2.minAreaRect(approx)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            # cv2.drawContours(frame,[box],0,(0,0,255),2)

            # (x, y) upper left, (x+w, y) upper right, (x, y+h) bottom left, (x+w, y+h) bottom right

            if first_frame_flag:
                first_frame_points = np.float32(
                    [[x, y], [x, y + h], [x + w, y + h], [x + w, y]]
                )
                first_frame_img = frame.copy()
                first_frame_flag = False
                roi = frame[y : y + h, x : x + w]

        # current_bb = np.float32([[x, y],[x+w, y+h],[x, y+h],[x+w, y]])
        if len(current_bb) == 4:
            dst_bb = np.float32(current_bb)
            prev_bb = np.float32(current_bb)
        else:
            dst_bb = prev_bb
        # cv2.circle(frame, tuple(dst_bb[0]), 8, (0, 0, 255), -1) #red - upper left
        # cv2.circle(frame, tuple(dst_bb[1]), 8, (0, 255, 0), -1) #green - bottom left
        # cv2.circle(frame, tuple(dst_bb[2]), 8, (255, 0, 0), -1) #blue - bottom right
        # cv2.circle(frame, tuple(dst_bb[3]), 8, (255, 255, 0), -1) #aqua - upper right

        orb = cv2.ORB_create()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = orb.detectAndCompute(first_frame_img, None)
        kp2, des2 = orb.detectAndCompute(frame, None)

        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match descriptors.
        matches = bf.match(des1, des2)

        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)

        # Draw first 10 matches.
        img3 = cv2.drawMatches(
            first_frame_img, kp1, frame, kp2, matches[:4], None, flags=2
        )

        # From DMatches to coordinates
        list_kp1 = [kp1[mat.queryIdx].pt for mat in matches]
        list_kp2 = [kp2[mat.trainIdx].pt for mat in matches]

        M, _ = cv2.findHomography(np.float32(list_kp1[:4]), np.float32(list_kp2[:4]))
        imageWarped = cv2.warpPerspective(frame, M, (frame_width, frame_height))

        # out.write(img3)
        cv2.imshow("Frame", frame)
        # cv2.imshow('Warped',imageWarped)

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
