from painting_retrieval import RetrieveClass
import cv2
import pandas as pd
import json


def localize():
    cap = cv2.VideoCapture("videos/VIRB0416.MP4")
    if cap.isOpened() == False:
        print("Error opening video stream or file")
    retrieve = RetrieveClass()
    museum_map = cv2.imread('resources/map.png')
    with open('resources/rooms_coordinates.json', 'r') as fp:
        room_coordinates = json.load(fp)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            paintings_ranked_list = retrieve.retrieval(frame.copy())
            best_match = max(paintings_ranked_list, key=paintings_ranked_list.get)
            if paintings_ranked_list[best_match] != 0:
                df = pd.read_csv('resources/data.csv')
                room = df[df.Image == best_match[-7:]].Room.item()
                start_point = (room_coordinates[str(room)][0], room_coordinates[str(room)][1])
                end_point = (room_coordinates[str(room)][2], room_coordinates[str(room)][3])
                color = (0, 0, 255)
                thickness = 5
                print('\nThe painting is in room ' + str(room))
                image = cv2.rectangle(museum_map.copy(), start_point, end_point, color, thickness)
                cv2.namedWindow("Map", cv2.WINDOW_KEEPRATIO)
                cv2.imshow("Map", image)
                cv2.resizeWindow("Map", int(image.shape[1] / 2), int(image.shape[0] / 2))
                cv2.waitKey()
                cv2.destroyWindow('Map')
            else:
                print("Can't determine the room")


if __name__ == '__main__':
    localize()
