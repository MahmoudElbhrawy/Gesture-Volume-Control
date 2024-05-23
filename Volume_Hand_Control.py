"""
Volume Hand Control
By: Mahmoud Elbhrawy, Eslam Ahmed, Moaaz Mohamed, Madlen Nady, Mohamed Hany
"""

import cv2
import time
import mediapipe as mp
import math
import numpy as np

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


class HandDetector:
        # the basic parameters required for Hands.py in mediapipe.
    def __init__(self, mode=False, max_hands=2, detection_confidence=0.5, tracking_confidence=0.5):
        # assigning variables initially with a value provided by the user
        # Setting the mode of operation (default to False, meaning static image mode).
        self.mode = mode
        # Maximum number of hands to detect
        self.max_hands = max_hands
        # Confidence threshold for hand detection
        self.detection_confidence = detection_confidence
        # Confidence threshold for hand tracking
        self.tracking_confidence = tracking_confidence
        # Importing the mediapipe hands module.
        self.mpHands = mp.solutions.hands
        # Creating a Hands object for hand detection
        self.hands = self.mpHands.Hands()
        # Importing the drawing utilities module from mediapipe
        self.mpDraw = mp.solutions.drawing_utils

    def find_hands(self, img, draw=True):
        # Convert the BGR image to RGB.
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # process the frame of image & give the results of detecting
        self.results = self.hands.process(img_rgb)
        # check if there is hand or not
        if self.results.multi_hand_landmarks:
            # Iterate through number of hands
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    # draw the 21 landmark for each Hand
                    # draw the connections between Landmarks
                    self.mpDraw.draw_landmarks(img, hand_landmarks, self.mpHands.HAND_CONNECTIONS)
        return img

    def find_position(self, img, hand_number=0, draw=True):
        # List to store landmark positions
        lm_list = []
        # check if there is hand or not
        if self.results.multi_hand_landmarks:
            # Get the landmarks for the specified hand number
            my_hand = self.results.multi_hand_landmarks[hand_number]
            # Iterate through index numbers for each Landmark
            for id, lm in enumerate(my_hand.landmark):
                # width , height , channels of our image (deminsions)
                h, w, c = img.shape
                # Calculate the coordinates of the landmark relative to the image size
                cx, cy = int(lm.x * w), int(lm.y * h)
                # Append landmark id and coordinates to the list
                lm_list.append([id, cx, cy])
                if draw:
                    # color and radius for landmarks
                    cv2.circle(img, (cx, cy), 4, (50, 130, 10), cv2.FILLED)
        return lm_list


def main():
    wCam, hCam = 640, 480  # Parameters of Camera (width & height)
    cap = cv2.VideoCapture(0)  # Using id 0 for VideoCapture

    # id number of width & height
    cap.set(3, wCam)
    cap.set(4, hCam)

    detector = HandDetector()  # Create object to call hand_detector

    pTime = 0  # Previous time

    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
        IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = interface.QueryInterface(IAudioEndpointVolume)
    # volume.GetMute()
    # volume.GetMasterVolumeLevel()
    volRange = volume.GetVolumeRange()  # (min : -65.25, max : 0.0,  0.03125)

    minVol = volRange[0]  # index 0 = -65.25
    maxVol = volRange[1]  # index 1 = 0.0

    vol = 0
    volBar = 400
    volPar = 0

    while True:
        success, img = cap.read()  # Check the success of the capture

        img = detector.find_hands(img)  # Send "findHands" in the img to give the hand

        lm_list = detector.find_position(img)  # landmark list (21)

        # print landmark list [4],[8]
        # if len(lm_list) != 0:
        # print(lm_list[4], lm_list[8])

        # Ensure lm_list has at least 9 landmarks
        if len(lm_list) >= 9:

            x1, y1 = lm_list[4][1], lm_list[4][2]
            x2, y2 = lm_list[8][1], lm_list[8][2]

            # calculate the centre between landmark [4],[8]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # circles at landmark[4],[8] and centre
            cv2.circle(img, (x1, y1), 7, (300, 300, 30), cv2.FILLED)
            cv2.circle(img, (x2, y2), 7, (300, 300, 30), cv2.FILLED)
            cv2.circle(img, (cx, cy), 7, (230, 68, 28), cv2.FILLED)

            # line between landmark[4],[8]
            cv2.line(img, (x1, y1), (x2, y2), (230, 68, 28), 3)

            # calculate length of the line between landmark [4],[8]
            length = math.hypot(x2 - x1, y2 - y1)

            # Hand Range from 30 to 250
            # Volume Range from -65.25 to 0

            # convert Hand Range into Volume Range
            vol = np.interp(length, [30, 300], [minVol, maxVol])
            # convert Hand Range into Volume Range Bar in the img
            volBar = np.interp(length, [30, 300], [400, 150])
            volPar = np.interp(length, [50, 300], [0, 100])
            # print(int(length) , vol)

            # set Hand Range to the PC volume
            volume.SetMasterVolumeLevel(vol, None)

            # change the center circle color when the length is too small
            if length < 30:
                cv2.circle(img, (cx, cy), 7, (0, 0, 255), cv2.FILLED)

        cv2.rectangle(img, (50, 150), (85, 400), (201, 47, 8), 3)
        cv2.rectangle(img, (50, int(volBar)), (85, 400), (230, 68, 28), cv2.FILLED)

        cv2.putText(img, f'{int(volPar)} %', (40, 450), cv2.FONT_ITALIC, 0.8, (0, 0, 0), 2)


        cTime = time.time()        # Current time
        fps = 1 / (cTime - pTime)  # Frame per second
        pTime = cTime

        # Print FPS in the img
        cv2.putText(img, f'FPS: {int(fps)}', (510, 50), cv2.FONT_ITALIC, 0.8, (0, 0, 0), 2)

        cv2.imshow("Image", img)
        cv2.waitKey(1)  # 1ms delay


if __name__ == "__main__":
    main()
