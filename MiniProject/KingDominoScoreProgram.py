import random
import CrownFinding as crownFinder
import cv2
import os
import numpy as np
import ImageSlicer as slice
import HistComparison as hist
import MatrixCreator as matrix


areaBrickDict = ["forest", "grassplane", "lake", "mine", "swamp", "wheat", "unknown"]

brickDict = {"BrickType": str,
             "Crowns": np.uint8(0),
             "checked": False,
             "ImageID": np.uint8(0)}

propertyDict = {"BrickType": "",
                "Count": np.uint8(0),
                "Crowns": np.uint8(0)}

image = cv2.imread("King Domino dataset/FullBoardsTestData/65.jpg")

'''alpha = 2
# control brightness by 50
beta = 0
image = cv2.convertScaleAbs(imageRaw, alpha=alpha, beta=beta)'''
#;

kernel_size_x = image.shape[0] // 5
kernel_size_y = image.shape[1] // 5

kernel_radius_x = kernel_size_x // 2
kernel_radius_y = kernel_size_y // 2

croppedImages = []
brickTypes = []


croppedImages = slice.slice_image(image)

#input_image_folder()

for croppedImage in croppedImages:
    brickTypes.append(hist.classify_brick(croppedImage))

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
        if not matrix[y, x]["checked"] : #and matrix[y, x]["BrickType"] != -1
            brickType = matrix[y, x]["BrickType"]
            count, crowns = dfs(matrix, x, y, brickType, 0, 0)
            matrix[y, x]["checked"] = True

            prop = propertyDict.copy()
            prop["BrickType"] = brickType
            prop["Count"] = count
            prop["Crowns"] = crowns
            properties.append(prop)


def calculate_final_score(properties):
    final_score = 0
    for prop in properties:
        final_score += prop["Count"] * prop["Crowns"]

    return final_score

print(properties)
print(len(properties))
cv2.imshow("image", image)
print(f"The final score is: {calculate_final_score(properties)}")


cv2.waitKey(0)
cv2.destroyAllWindows()
