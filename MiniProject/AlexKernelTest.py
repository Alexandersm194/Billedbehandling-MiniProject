import random
import CrownFinding as crownFinder
import cv2
import os
import numpy as np
import EvaluationScript as eval
import ImageSlicer as slice
import HistComparison as hist
import MatrixCreator as matrix

def input_image_folder():
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
        input_image_folder()




areaBrickDict = {"forest": 0,
                  "swamp": 1,
                  "mine": 2,
                  "grassplane": 3,
                  "lake": 4,
                  "wheat": 5,
                 "unknown": 6}

brickDict = {"BrickType": np.uint8(0),
             "Crowns": np.uint8(0),
             "checked": False,
             "ImageID": np.uint8(0)}

propertyDict = {"BrickType": "",
                "Count": np.uint8(0),
                "Crowns": np.uint8(0)}

image = cv2.imread("King Domino dataset/FullBoardsTestData/9.jpg")


kernel_size_x = image.shape[0] // 5
kernel_size_y = image.shape[1] // 5

kernel_radius_x = kernel_size_x // 2
kernel_radius_y = kernel_size_y // 2

croppedImages = []
preImages = []
fileNames = []
brickTypes = []


croppedImages = slice.slice_image(image)

print(len(croppedImages))

input_image_folder()

for croppedImage in croppedImages:
    brickTypes.append(hist.classify_brick(croppedImage))
    '''if comparisonVar > 0.55:
        brickTypes.append(fileNames[wonImage])
        print(f"Won image: {wonImage}")
    else:
        brickTypes.append(None)
        print("No Won image")'''

matrix = matrix.createMatrix(brickDict, brickTypes, areaBrickDict)



def calculate_crowns_per_square(ImageID):
    return crownFinder.crownEdges(croppedImages[ImageID])


def dfs(matrix, x, y, BrickType, in_count, crowns):
    # Check boundary conditions and color match
    if (x < 0 or x >= 5 or y < 0 or y >= 5 or matrix[y, x]["checked"] == True
            or matrix[y, x]["BrickType"] != BrickType):
        return in_count, crowns

    matrix[y, x]["checked"] = True

    in_count = in_count + 1

    crowns = crowns + calculate_crowns_per_square(matrix[y, x]["ImageID"])

    # Visit all adjacent pixels
    in_count, crowns = dfs(matrix, x + 1, y, BrickType, in_count, crowns)
    in_count, crowns = dfs(matrix, x - 1, y, BrickType, in_count, crowns)
    in_count, crowns = dfs(matrix, x, y + 1, BrickType, in_count, crowns)
    in_count, crowns = dfs(matrix, x, y - 1, BrickType, in_count, crowns)

    return in_count, crowns


properties = []

for y in range(matrix.shape[0]):
    for x in range(matrix.shape[1]):
        if not matrix[y, x]["checked"] and matrix[y, x]["BrickType"] != -1:
            brickType = matrix[y, x]["BrickType"]
            count, crowns = dfs(matrix, x, y, brickType, 0, 0)
            matrix[y, x]["checked"] = True

            prop = propertyDict.copy()
            prop["BrickType"] = brickType
            prop["Count"] = count
            prop["Crowns"] = crowns
            properties.append(prop)


def calculate_final_score():
    final_score = 0
    for prop in properties:
        final_score += prop["Count"] * prop["Crowns"]

    return final_score

print(properties)
print(len(properties))
cv2.imshow("image", image)
print(f"The final score is: {calculate_final_score()}")

#print(eval.evaluate(matrix))

cv2.waitKey(0)
cv2.destroyAllWindows()
