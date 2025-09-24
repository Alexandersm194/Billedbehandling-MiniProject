import random

import cv2
import os
import numpy as np


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


def calculate_hue_hist(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue_channel = hsv_image[:, :, 0]
    return cv2.calcHist([hue_channel], [0], None, [255], [0, 255])


areaBrickDict = {"forestProb.jpg": 0,
                  "GreyProb.jpg": 1,
                  "MineProb.jpg": 2,
                  "planeProb.jpg": 3,
                  "WaterProb.jpg": 4,
                  "YellowProb.jpg": 5}

brickDict = {"BrickType": np.uint8(0),
             "Crowns": np.uint8(0),
             "checked": False,
             "ImageID": np.uint8(0)}

propertyDict = {"BrickType": "",
                "Count": np.uint8(0),
                "Crowns": np.uint8(0)}

image = cv2.imread("King Domino dataset//Cropped and perspective corrected boards//1.jpg")

kernel_size_x = image.shape[0] // 5
kernel_size_y = image.shape[1] // 5

kernel_radius_x = kernel_size_x // 2
kernel_radius_y = kernel_size_y // 2

croppedImages = []
preImages = []
fileNames = []
brickTypes = []

#matrix = np.zeros((5, 5), dtype=type(brickDict))
matrix = np.empty((5, 5), dtype=object)
for i in range(5):
    for j in range(5):
        matrix[i, j] = brickDict.copy()

for y in range(5):
    for x in range(5):
        pixelValues = [0, 0]
        if(y == 0 and x == 0):
            pixelValues = [kernel_radius_y, kernel_radius_x]
            croppedImages.append(image[0:kernel_size_y, 0:kernel_size_x])
        elif(y == 0 and x != 0):
            pixelValues = [kernel_radius_y, kernel_radius_x + kernel_size_x * x]
            croppedImages.append(image[0:kernel_size_y, kernel_size_x * x:kernel_size_x * x + kernel_size_x])
        elif(y != 0 and x == 0):
            pixelValues = [kernel_radius_y + kernel_size_y * y, kernel_radius_x]
            croppedImages.append(image[kernel_size_y * y:kernel_size_y * y + kernel_size_y, 0:kernel_size_x])
        elif(y != 0 and x != 0):
            pixelValues = [kernel_radius_y + kernel_size_y * y, kernel_radius_x + kernel_size_x * x]
            croppedImages.append(image[kernel_size_y * y:kernel_size_y * y + kernel_size_y, kernel_size_x * x:kernel_size_x * x + kernel_size_x])


input_image_folder()

for croppedImage in croppedImages:
    comparisonVar = 0
    wonImage = 0
    comparisonMethod = 0

    croppedHist = calculate_hue_hist(croppedImage)
    for i, compareImage in enumerate(preImages):
        comparedHist = calculate_hue_hist(compareImage)
        cor = cv2.compareHist(croppedHist, comparedHist, comparisonMethod)

        if i == 0:
            comparisonVar = cor

        elif cor > comparisonVar:
            comparisonVar = cor
            wonImage = i

    brickTypes.append(fileNames[wonImage])

croppedImgIndex = 0
for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        var = areaBrickDict.get(brickTypes[croppedImgIndex])
        matrix[i, j]["BrickType"] = var
        matrix[i, j]["ImageID"] = croppedImgIndex
        croppedImgIndex += 1


def calculate_crowns_per_square(ImageID):
    return random.randint(0, 2)


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
        if not matrix[y, x]["checked"]:
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
cv2.imshow("image", image)
print(f"The final score is: {calculate_final_score()}")

cv2.waitKey(0)
cv2.destroyAllWindows()
