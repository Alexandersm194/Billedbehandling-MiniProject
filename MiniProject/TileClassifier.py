import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def input_trainingdata(dir):
    array = []
    if os.path.isdir(dir):
        for file in os.listdir(dir):
            full_path = os.path.join(dir, file)
            img = cv2.imread(full_path)
            if img is not None:
                array.append(img)
    else:
        print("This is not a functional path!")

    return array


def calculate_hue_hist(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue_channel = hsv_image[:, :, 0]
    return cv2.calcHist([hue_channel], [0], None, [255], [0, 255])


tileType = ["forest", "grassplane", "lake", "mine", "swamp", "wheat", "unknown"]

trainingDataGroups = {
    "forest": input_trainingdata("dataset//forest"),
    "grassplane": input_trainingdata("dataset//grasslands"),
    "lake": input_trainingdata("dataset//lake"),
    "mine": input_trainingdata("dataset//mine"),
    "swamp": input_trainingdata("dataset//swamp"),
    "wheat": input_trainingdata("dataset//wheat"),
    "unknown": input_trainingdata("dataset//unknown")
}

trainingDataGroupsHist = {
    "forest": [],
    "grassplane": [],
    "lake": [],
    "mine": [],
    "swamp": [],
    "wheat": [],
    "unknown": []
}

for type in tileType:
    for i, compareImg in enumerate(trainingDataGroups[type]):
        blur = cv2.blur(compareImg, (23, 23))
        sharpenedImage = cv2.addWeighted(compareImg, 1.7, blur, -0.8, 0)
        trainingDataGroupsHist[type].append(calculate_hue_hist(sharpenedImage))


def classify_tile(orgImage):
    compVar = float("inf")
    predictType = 0
    compMethod = cv2.HISTCMP_CHISQR

    croppedHist = calculate_hue_hist(orgImage)

    for i, type in enumerate(tileType):
        for compareImg in trainingDataGroupsHist[type]:
            dist = cv2.compareHist(croppedHist, compareImg, compMethod)

            if dist < compVar:
                compVar = dist
                predictType = i

    return tileType[predictType]
