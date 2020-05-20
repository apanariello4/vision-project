import cv2

video = "VIRB0391"
folder = "000"
cap = cv2.VideoCapture("videos/VIRB0391.MP4")

if cap.isOpened() == False:
    print("Error opening video stream or file")
i = 0
while cap.isOpened():
    ret, frame = cap.read()
    i += 1
    if ret == True:
        cv2.imwrite(
            f'/home/nello/Desktop/vision-project/images/{video}-{str(i)}.png', frame)
        print(f"FRAME {str(i)}")
        cap.set(1, i*15)
    else:
        break

cap.release()
