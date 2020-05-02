import cv2
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import painting_detection
from utils import *


def init():
    """
    Initialization function
        :return: all we need
    """
    fm = cv2.ORB_create()
    return fm, cv2.BFMatcher(cv2.NORM_HAMMING), cv2.VideoCapture(
        "videos/VIRB0392.mp4"), 0, load_keypoints(compute_and_write=False, matcher=fm)


def retrieval():
    brisk, bf, cap, frame_number, (images_db, keypoints_db, descriptors_db) = init()

    while cap.isOpened():
        start = time.time()

        matched_collage = np.array([])
        good_global = 0
        ret, frame = cap.read()
        images_ranked_list = {}
        contours, _ = painting_detection.contours(frame, adaptive=False)
        frame_number += 1

        print("\n############  FRAME N°" + str(frame_number) + "  ############")
        if len(contours) != 0:

            img_frame = get_painting_from_roi(contours, frame)

            kp_frame, desc_frame = brisk.detectAndCompute(img_frame, None)

            for file in glob.glob("paintings_db/*.png"):

                matches = bf.knnMatch(desc_frame, descriptors_db[file], k=2)

                good = get_good_matches(matches)

                collage = cv2.drawMatchesKnn(
                    img_frame,
                    kp_frame,
                    images_db[file],
                    keypoints_db[file],
                    good,
                    None,
                    flags=2,
                )

                images_ranked_list[file] = np.asarray(good).shape[0]
                if int(images_ranked_list[file]) > int(good_global):
                    good_global = images_ranked_list[file]
                    matched_collage = collage

                check_match(collage)
                if cv2.waitKey(10) & 0xFF == ord("q"):
                    break

            end = time.time()
            print("\nTime to search the matched image: " + "%.2f" % (
                    end - start) + " seconds\n")
            show_match(matched_collage)
            print_ranked_list(images_ranked_list)


if __name__ == '__main__':
    retrieval()
