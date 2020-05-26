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

    def is_facing_camera(self, person_roi) -> bool:
        print("[INFO] Performing face detection")

        start = time.time()
        face = []
        gray_person_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
        face = self.face_cascade.detectMultiScale(gray_person_roi, 1.3, 5)
        if len(face) != 0:
            self.draw_face(person_roi, face[0])
            (x, y, w, h) = face[0]
            face_roi = gray_person_roi[y:y + h, x:x + w]
            eyes = self.eye_cascade.detectMultiScale(face_roi)
            end = time.time()
            print("[INFO] Face found. Time to detect the face: " + "%.2f" % (
                    end - start) + " seconds")
            if len(eyes) != 0:
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(person_roi, (ex + x, ey + y), (ex + ew + x, ey + eh + y), (0, 255, 0), 2)
                self.show_img(person_roi)
                return True
            else:
                self.show_img(person_roi)
                return False
        else:
            end = time.time()
            print("[INFO] No face found. Time to detect the face: " + "%.2f" % (
                    end - start) + " seconds")
            return False

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
        cv2.waitKey(0)
        cv2.destroyAllWindows()
