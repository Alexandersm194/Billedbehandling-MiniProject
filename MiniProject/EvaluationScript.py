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

def evaluate(programMatrix):
    programResults = []
    for i, row in enumerate(programMatrix):
        for j, img in enumerate(row):
            programResults.append(img[i, j]["BrickType"])

    for i, truth in enumerate(groundtruth):
        if truth == programResults[i]:
            brick_confusion_matrix[programResults[i]][truth] += 1


    return brick_confusion_matrix

