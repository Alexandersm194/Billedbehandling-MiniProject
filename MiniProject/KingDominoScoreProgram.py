import random
import CrownFinding as cf
import cv2
import os
import numpy as np
import ImageSlicer as slice
import TileClassifier as tc
import MatrixCreator as matrix
import ScoreCalculator as sc


image = cv2.imread("King Domino dataset/FullBoardsTestData/65.jpg")

croppedImages = []
tileTypes = []
numberOfCrowns = []

croppedImages = slice.slice_image(image)

for croppedImage in croppedImages:
    tileTypes.append(tc.classify_tile(croppedImage))
    numberOfCrowns.append(cf.find_crowns(croppedImage))

full_board_matrix = matrix.createMatrix(tileTypes, numberOfCrowns)

final_score = sc.counter(full_board_matrix)

cv2.imshow("image", image)
print(f"The final score is: {final_score}")

cv2.waitKey(0)
cv2.destroyAllWindows()
