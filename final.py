from cv2 import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
offset = 20                              # def offset bcz complete img not shown in ImageCrop
imgSize = 300

# To save images
folder = "Data/B"
counter = 0
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

while True:

    success, img = cap.read()
    imgOutput = img.copy()                # creating a copy of the img to edit and add new things
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]                  # detecting hand object
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
            prediction, index = classifier.getPrediction(imgWhite,draw=False)
            print(prediction, index)

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
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            print(prediction, index)

        cv2.rectangle(imgOutput, (x - offset, y - offset-50), (x- offset + 90, y - offset-50 + 50), (94, 255, 5), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y-25), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
        cv2.rectangle(imgOutput, (x-offset, y-offset), (x+w+offset, y+h+offset), (94, 255, 5), 4)
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("img", imgOutput)
    cv2.waitKey(1)
