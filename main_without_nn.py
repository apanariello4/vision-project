from cv2 import cv2
import painting_detection
from utils import show_img
from ccl import image_segmentation

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture("videos/Sample video.MP4")

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

first_frame_flag = True

# Check if camera opened successfully
if cap.isOpened() == False:
    print("Error opening video stream or file")

# Read until video is completed
while cap.isOpened():

    ret, frame = cap.read()
    scale_percent = 20  # percent of original size
    width = int(frame_width * scale_percent / 100)
    height = int(frame_height * scale_percent / 100)
    dim = (width, height)
    # resize image
    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    if ret == True:
        contours, hierarchy = painting_detection.contours(image_segmentation(frame), adaptive=True)
        if len(contours) != 0:
            painting_detection.dram_multiple_contours(frame, contours, approximate=False)
        show_img("Frame", frame)

    # Break the loop
    else:
        break

# When everything done, release the video capture and writer objects
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
