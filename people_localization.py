import cv2
import pandas as pd
import json
import os
from scripts.write_map_coordinates import write_json_coordinates
import time


class LocalizeClass:

    def __init__(self):
        self.paintings_ranked_list = {}
        self.museum_map = cv2.imread('resources/map.png')
        self.df = pd.read_csv('resources/data.csv')
        self.rect_color = (0, 0, 255)
        self.rect_thickness = 5
        self.room_coordinates = {}
        self.loaded_coordinates = False
        self.load_room_coordinates()
        print("[INFO] Ready for people localization")
        print("___________________________________")

    def load_room_coordinates(self) -> None:
        """
        It loads the room coordinates from a file. If the file is not in the specified path,
        it runs a script to generate it
        """
        start = time.time()
        print("[INFO] Searching for room coordinates")
        if os.path.exists('resources/rooms_coordinates.json') and os.path.exists('resources/rooms_coordinates.json'):
            print("[SUCCESS] File found")
            with open('resources/rooms_coordinates.json', 'r') as fp:
                self.room_coordinates = json.load(fp)
            self.loaded_coordinates = True
        else:
            print("[INFO] File not found, running write_map_coordinates.py to generate the file")
            write_json_coordinates()
            if os.path.exists('resources/rooms_coordinates.json') and os.path.exists(
                    'resources/rooms_coordinates.json'):
                with open('resources/rooms_coordinates.json', 'r') as fp:
                    self.room_coordinates = json.load(fp)
                self.loaded_coordinates = True
                print("[SUCCESS] Rooms coordinates successfully generated and loaded into memory")
            else:
                print("[ERROR] There is an error with write_map_coordinates.py script")
        end = time.time()
        print("[INFO] Loading time: " + "%.4f" % (end - start) + " seconds")

    def localize(self, ranked_list):
        """
        It takes a ranked list of paintings and determine the room in which the painting is in
            :param ranked_list: ranked list of matches for each painting
        """
        print("[INFO] Performing people localization")
        start = time.time()

        if not self.loaded_coordinates:
            self.load_room_coordinates()

        self.paintings_ranked_list = ranked_list
        best_match = max(self.paintings_ranked_list, key=self.paintings_ranked_list.get)
        if self.paintings_ranked_list[best_match] > 0:
            room = self.df[self.df.Image == best_match[-7:]].Room.item()
            start_point = (self.room_coordinates[str(room)][0], self.room_coordinates[str(room)][1])
            end_point = (self.room_coordinates[str(room)][2], self.room_coordinates[str(room)][3])
            end = time.time()

            print('[SUCCESS] The painting is in room ' + str(room) + ". Time to detect the face: " + "%.2f" % (
                    end - start) + " seconds")
            self.show_map(
                cv2.rectangle(self.museum_map.copy(), start_point, end_point, self.rect_color, self.rect_thickness))
            return room
        else:
            print("[FAIL] Can't determine the room")
            return None

    def show_map(self, img):
        cv2.namedWindow("Map", cv2.WINDOW_KEEPRATIO)
        cv2.imshow("Map", img)
        cv2.resizeWindow("Map", int(img.shape[1] / 2), int(img.shape[0] / 2))
        cv2.waitKey()
        cv2.destroyWindow('Map')
