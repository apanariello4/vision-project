import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name


videolist = glob.glob('videos/*.mp4')

for video in videolist:
    cap = cv2.VideoCapture(video)
    print("STIAMO APRENDO IL VIDEO : ",video)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    codec = cv2.VideoWriter_fourcc(*"DIVX")
    fps = 30
    out = cv2.VideoWriter('output.avi',0, fps, (frame_width,frame_height))
    og = True

    # Check if camera opened successfully
    if (cap.isOpened()== False):
        print("Error opening video stream or file")

    good_global = 0
    name_painting = None

    time = 0
    time_max = 1000  # max seconds of video (10msec)
    time_iterate = 200

    # Read until video is completed
    finish = False

    while(cap.isOpened() and finish == False):

        #current position of the video in 10msec (Non in millisecondi maledetta reference)
        cap.set(1,time)
        ret, frame = cap.read()
        prev_img = None


        if ret == True:

            filelist = glob.glob('images/09*.png')

            if name_painting == None:

                img1 = frame
                    #img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                img1 = cv2.resize(img1, (960, 540))
                for fname in filelist:
                    img2 = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)

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
                    dim_attuale = np.asarray(good).shape[0]

                    if int(dim_attuale) > int(good_global):
                        good_global = dim_attuale
                        name_painting = fname
                    # cv.drawMatchesKnn expects list of lists as matches.

                    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


                    cv2.imshow('Frame',img3)

                    # Press Q on keyboard to  exit
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break

            # update and control time
            time += time_iterate

            #picture matched
            if name_painting:
                print("ecco la corrispondenza")
                img2 = cv2.imread(name_painting, cv2.IMREAD_GRAYSCALE)
                finish = True

                cv2.imshow('Frame', img2)
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break
            else:
                if time >= time_max:
                    print("nessuna corrispondenza")
                    finish = True
        # Break the loop
        else:
            break

    # When everything done, release the video capture and writer objects
    cap.release()
    out.release()

    # Closes all the frames
    cv2.destroyAllWindows()

print("FINE")

