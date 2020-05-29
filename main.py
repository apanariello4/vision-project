import os

import numpy as np
from cv2 import cv2

import painting_detection
from ccl import draw_components, image_segmentation
from darknet_yolo.darknet_pytorch import Darknet
from htrdc import HTRDC, undistort
from people_detection import DetectNet
from painting_rectification import RectifyClass
from painting_retrieval import RetrieveClass
from face_detection import FaceDetectionClass
from utils import resize_when_too_big, is_roi_outside_frame, equalize_luma

HTRDC_K_START = 0.0
HTRDC_K_END = 1e-4
HTRDC_N = 20
HTRDC_EPSILON = 1e-6
DEBUG = True


def compute_HTRDC(img, k):
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
    video_path = "videos/VIRB0392.MP4"
    # cap = cv2.VideoCapture("videos/VIRB0392.MP4")
    # cap.set(1, 700)
    cap = cv2.VideoCapture(video_path)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # codec = cv2.VideoWriter_fourcc(*"DIVX")
    # fps = 30
    # out = cv2.VideoWriter("output.avi", 0, fps, (frame_width, frame_height))

    k = None
    frame_number = 0

    detect = Darknet()
    retrieve = RetrieveClass()
    rectify = RectifyClass(retrieve)
    face_detection = FaceDetectionClass()

    # Check the extension of video, MOV files are rotated by -90°
    file_name, file_extension = os.path.splitext(video_path)

    # Check if camera opened successfully
    if cap.isOpened() == False:
        print("Error opening video stream or file")

    # Read until video is completed
    while cap.isOpened():

        ret, frame = cap.read()

        if ret == True:
            frame_number += 1
            print(f'############  FRAME N° {frame_number}  ############')

            if file_extension.upper() == "MOV":
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            # bb_frame is used for drawing, frame is clean
            bb_frame = frame.copy()
            detections_list = detect.yolo_detection(bb_frame, draw=True)

            for detection in detections_list:
                if detection[0] == 'painting' and not is_roi_outside_frame(
                        frame_width, frame_height, *detection[2:6]):

                    left = detection[2]  # x
                    top = detection[3]  # y
                    right = detection[4]  # x + w
                    bottom = detection[5]  # y + h

                    painting = frame[int(top):int(bottom),
                                     int(left):int(right)]

                    print("[INFO] Painting detected")
                    cv2.rectangle(bb_frame, (left, bottom+10),
                                  (left+50, bottom+10), )
                    cv2.imshow("frame", bb_frame)
                    cv2.waitKey()

                    rectify.rectify(painting)
                if detection[0] == 'painting' and len(detections_list) == 1:
                    # It founds only one painting, the roi can be outside frame

                    left = max(0, detection[2])  # x
                    top = max(0, detection[3])  # y
                    right = min(detection[4], frame_height)  # x + w
                    bottom = min(detection[5], frame_width)  # y + h

                    painting = frame[int(top):int(bottom),
                                     int(left):int(right)]

                    print("[INFO] Painting detected")

                    rectify.rectify(painting)

                elif detection[0] == 'person':
                    # FACE DETECTION
                    left = detection[2]
                    right = detection[4]
                    top = detection[3]
                    bottom = detection[5]

                    # TO DO AGGIUSTARE IL ROI
                    print("[INFO] Person detected")

                    person_roi = frame[int(top * 0.75):int(bottom), left:right]

                    if not face_detection.is_facing_camera(person_roi, True):
                        paintings_detections = [
                            element for element in detections_list if element[0] == 'painting']
                        if face_detection.is_facing_paintings(detection, paintings_detections):
                            cv2.rectangle(
                                bb_frame, (left, bottom+10), (left+50, bottom+10))

                    txt_color = (255, 255, 255)
                    label = "The person is facing a painting"
                    label_size, base_line = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    bottom = min(bottom, label_size[1])
                    cv2.rectangle(bb_frame, (left, bottom - round(1.5 * label_size[1])),
                                  (left + round(1.5 * label_size[0]), bottom + base_line), color=(0, 0, 0), thickness=cv2.FILLED)
                    cv2.putText(bb_frame, label, (left, bottom), cv2.FONT_HERSHEY_SIMPLEX,
                                0.75, color=txt_color, thickness=2)

                    # TO DO gestire quando è rivolto a paintings

                elif detection[0] == 'statue':
                    # TO DO
                    pass
                else:
                    print("[INFO] Nothing detected")
                    pass

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord("q"):
                break

        # Break the loop
        else:
            break

    # When everything done, release the video capture and writer objects
    cap.release()
    # out.release()

    # Closes all the frames
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
