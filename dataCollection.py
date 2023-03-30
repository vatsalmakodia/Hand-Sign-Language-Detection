from cv2 import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20                              # def offset bcz complete img not shown in ImageCrop
imgSize = 300

# To save images
folder = "Data/9"
counter = 0

while True:

    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]                # detecting hand object
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255      # generate a white image
        imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]      # dimensions of cropped img in the form of matrix

        imgCropShape = imgCrop.shape       # overlay on top of white img
        # matrix consisting 0 to imgCropShape height-->0, 0 to imgCropShape width-->1

        aspectRatio = h/w

        # For height to always be 300/imgSize
        if aspectRatio > 1:
            k = imgSize/h
            wCal = math.ceil(k*w)               # ceil is just like round off
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            # To bring img in centre
            wGap = math.ceil((imgSize-wCal)/2)
            imgWhite[:, wGap:wCal + wGap] = imgResize      # overlay on top of white img
            # matrix consisting of [h,w,channel], [start:end]

        # For width to always be 300/imgSize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)  # hCal = height calculated
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            # To bring img in centre
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize  # overlay on top of white img
            # matrix consisting of [h,w,channel], [start:end]

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("img", img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f"{folder}/Image_{time.time()}.jpg", imgWhite)
        print(counter)
