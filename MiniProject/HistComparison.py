import cv2
import numpy as np
import os

preImages = []
fileNames = []

imageDir = "AreaBricks"

if os.path.isdir(imageDir):
    for file in os.listdir(imageDir):
        full_path = os.path.join(imageDir, file)
        img = cv2.imread(full_path)
        if img is None:
            print(f"Could not load image: {full_path}")
        else:
            print(f"Image loaded successfully: {full_path}")
            preImages.append(img)
            fileNames.append(file)
else:
    print("This is not a funtional path!")

def stuff(dir):
    array = []
    if os.path.isdir(dir):
        for file in os.listdir(dir):
            full_path = os.path.join(dir, file)
            img = cv2.imread(full_path)
            if img is not None:
                array.append(img)
    else:
        print("This is not a funtional path!")

    return array

trainingDataGroups = {
    "forest": stuff("dataset//forest"),
    "grassplane": stuff("dataset//grasslands"),
    "lake": stuff("dataset//lake"),
    "mine": stuff("dataset//mine"),
    "swamp": stuff("dataset//swamp"),
    "wheat": stuff("dataset//wheat"),
    "unknown": stuff("dataset//unknown")
}

brickType = ["forest", "grassplane", "lake", "mine", "swamp", "wheat", "unknown"]


def calculate_hue_hist(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue_channel = hsv_image[:, :, 0]
    return cv2.calcHist([hue_channel], [0], None, [255], [0, 255])


def classify_brick(orgImage):
    comparisonVar = float("inf")  # start high because smaller is better
    predictType = 0
    comparisonMethod = cv2.HISTCMP_CHISQR  # use Chi-square distance

    croppedHist = calculate_hue_hist(orgImage)

    for i, type in enumerate(brickType):
        for compareImg in trainingDataGroups[type]:
            comparedHist = calculate_hue_hist(compareImg)
            dist = cv2.compareHist(croppedHist, comparedHist, comparisonMethod)

        # Chi-square: smaller = more similar
            if dist < comparisonVar:
                comparisonVar = dist
                predictType = i

    return brickType[predictType]

    '''
    comparisonVar = 0
    wonImage = 0
    comparisonMethod = 0

    croppedHist = calculate_hue_hist(orgImage)
    for i, compareImage in enumerate(preImages):
        comparedHist = calculate_hue_hist(compareImage)
        cor = cv2.compareHist(croppedHist, comparedHist, comparisonMethod)

        if i == 0:
            comparisonVar = cor

        elif cor > comparisonVar:
            comparisonVar = cor
            wonImage = i
    return fileNames[wonImage]'''