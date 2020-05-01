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

def erosion_dilation(img):
    kernel = np.ones((5, 5), np.uint8)
    img_erosion = cv2.erode(img, kernel, iterations=1)
    dilation = cv2.dilate(img_erosion, kernel, iterations=1)
    img_erosion = cv2.erode(dilation, kernel, iterations=1)
    return img_erosion





def labeling(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #thresh = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,13,3) #11,2 at the beginning
    medianBlur = cv2.medianBlur(thresh, 5)
    post_proc = erosion_dilation(medianBlur)
    num_labels, labels_im = cv2.connectedComponentsWithAlgorithm(post_proc,connectivity=8,ltype=cv2.CV_32S,ccltype=cv2.CCL_GRANA)
    labeled_image = draw_components(labels_im)

    return labeled_image
