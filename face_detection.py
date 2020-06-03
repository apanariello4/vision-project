import cv2
import numpy as np
import time


class FaceDetectionClass():

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            'venv/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier('venv/lib/python3.7/site-packages/cv2/data/haarcascade_eye.xml')
        print("[INFO] Ready for face detection")
        print("___________________________________")

    def is_facing_camera(self, person_roi, draw=False) -> bool:
        print("[INFO] Performing face detection")

        start = time.time()
        face = []
        gray_person_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
        face = self.face_cascade.detectMultiScale(gray_person_roi, 1.3, 5)
        if len(face) != 0:
            if draw:
                self.draw_face(person_roi, face[0])
            (x, y, w, h) = face[0]
            face_roi = gray_person_roi[y:y + h, x:x + w]
            eyes = self.eye_cascade.detectMultiScale(face_roi)
            end = time.time()
            if len(eyes) != 0:
                if draw:
                    for (ex, ey, ew, eh) in eyes:
                        cv2.rectangle(person_roi, (ex + x, ey + y), (ex + ew + x, ey + eh + y), (0, 255, 0), 2)
                print("[INFO] The person is facing the camera. Time to detect the face: " + "%.2f" % (
                        end - start) + " seconds")
                self.show_img(person_roi)
                return True
            else:
                print("[INFO] The person is not facing the camera. Time to detect the face: " + "%.2f" % (
                        end - start) + " seconds")
                self.show_img(person_roi)
                return False
        else:
            end = time.time()
            print("[INFO] The person is not facing the camera. Time to detect the face: " + "%.2f" % (
                    end - start) + " seconds")
            return False

    def is_facing_paintings(self, person_detection, paintings_detections, draw=False):
        if len(paintings_detections) <= 0:
            print("[INFO] There are no paintings near the person")
            print("[INFO] The person is not facing any paintings")
            return False
        start = time.time()

        bb_person_left = person_detection[2]
        bb_person_right = person_detection[4]
        bb_person_top = person_detection[3]
        bb_person_bottom = person_detection[5]
        person_rect = [bb_person_top, bb_person_left, bb_person_bottom, bb_person_right]
        is_overlapping = False
        for painting_detection in paintings_detections:
            bb_painting_left = painting_detection[2]
            bb_painting_right = painting_detection[4]
            bb_painting_top = painting_detection[3]
            bb_painting_bottom = painting_detection[5]
            painting_rect = [bb_painting_top, bb_painting_left, bb_painting_bottom, bb_painting_right]
            is_overlapping = self.is_overlapping(person_rect, painting_rect)
            if is_overlapping:
                break
        end = time.time()
        if is_overlapping:
            print("[INFO] The person is facing at least one painting")
            print("[INFO] Time to check if the person if facing a painting: " + "%.3f" % (
                    end - start) + " seconds")
            return True
        else:
            print("[INFO] The person is not facing any paintings")
            print("[INFO] Time to check if the person if facing a painting: " + "%.3f" % (
                    end - start) + " seconds")
            return False

    def is_overlapping(self, rect1, rect2):
        min_y1, min_x1, max_y1, max_x1, = rect1
        min_y2, min_x2, max_y2, max_x2 = rect2
        if min_x1 > max_x2 or max_x1 < min_x2:
            return False
        if min_y1 > max_y2 or max_y1 < min_y2:
            return False
        return True

    def draw_face(self, img, face):
        (x, y, w, h) = face
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    def draw_eyes(self, img, eyes):
        ex, ey, ew, eh = eyes
        cv2.rectangle(img, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    def detect_faces_and_draw(self, person_roi) -> None:
        gray = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            person_roi = cv2.rectangle(person_roi, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = person_roi[y:y + h, x:x + w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        self.show_img(person_roi)

    def show_img(self, img):
        """
        If there is a matched image, the function shows it on screen
            :param img: image to show on screen
        """
        cv2.namedWindow("Face detection", cv2.WINDOW_KEEPRATIO)
        cv2.imshow("Face detection", img)
        cv2.resizeWindow("Face detection", int(img.shape[1] / 2), int(img.shape[0] / 2))
        cv2.waitKey()
        cv2.destroyAllWindows()
