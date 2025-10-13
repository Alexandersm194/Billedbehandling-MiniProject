import cv2
import numpy as np

brick_confusion_matrix = [["", "forest", "grasslands", "wheat", "swamp", "mine", "lake", "unknown"],
                          ["forest", 0, 0, 0, 0, 0, 0, 0],
                          ["grasslands", 0, 0, 0, 0, 0, 0, 0],
                          ["wheat", 0, 0, 0, 0, 0, 0, 0],
                          ["swamp", 0, 0, 0, 0, 0, 0, 0],
                          ["mine", 0, 0, 0, 0, 0, 0, 0],
                          ["lake", 0, 0, 0, 0, 0, 0, 0],
                          ["unknown", 0, 0, 0, 0, 0, 0, 0]]

brick_index = {
    "forest": 1,
    "grasslands": 2,
    "wheat": 3,
    "swamp": 4,
    "mine": 5,
    "lake": 6,
    "unknown": 7
}

crown_confusion_matrix = [["", "Crown found", "Crown not found"],
                          ["Crown", 0, 0],
                          ["No Crown", 0, 0]]
groundtruth = []

with open("GroundTruth//BrickGroundTruth.txt") as f:
  for x in f:
    groundtruth.append(x)

def evaluate(programMatrix):
    programResults = []
    for i, row in enumerate(programMatrix):
        for j, img in enumerate(row):
            programResults.append(programMatrix[i,j]["BrickType"])

    print(len(programResults))
    for x, truth in enumerate(groundtruth):
        if truth == programResults[x]:
            brick_confusion_matrix[programResults[x]][truth] += 1


    return brick_confusion_matrix

