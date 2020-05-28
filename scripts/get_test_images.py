import glob
import random
import os

import cv2

videos_path = "/home/nello/Desktop/Project material/videos"
output_path = "/home/nello/Desktop/vision-project/images"

videos_list = glob.glob(f'{videos_path}/**/*.*', recursive=False)

videos_sample = random.sample(videos_list, 50)

for video in videos_sample:
    cap = cv2.VideoCapture(video)
    file_path, file_extension = os.path.splitext(video)
    file_name = os.path.basename(file_path)

    # get total number of frames
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    my_frame = int(total_frames / 4)

    for i in range(1, 4):
        frame_number = i * my_frame
        # check for valid frame number
        if frame_number >= 0 & frame_number <= total_frames:
            # set frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        else:
            break

        ret, frame = cap.read()
        if file_extension.upper() == "MOV":
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(
            f'{output_path}/{file_name}-frame-{frame_number}.png', frame)
        print(f"Wrote {file_name}-frame-{frame_number}.png")

        # cv2.waitKey()
        # if cv2.waitKey(20) & 0xFF == ord('q'):
        #     break

    cv2.destroyAllWindows()
