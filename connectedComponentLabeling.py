import cv2
import numpy as np


def draw_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0

    cv2.imshow('labeled.png', labeled_img)
    cv2.waitKey()
    return labeled_img

def labeling(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    blur = cv2.blur(thresh, (5, 5))
    num_labels, labels_im = cv2.connectedComponents(thresh)
    labeled_image = draw_components(labels_im)
    return labeled_image
