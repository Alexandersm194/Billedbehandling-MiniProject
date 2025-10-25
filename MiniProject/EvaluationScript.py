import math
import cv2
import os
import numpy as np
import ImageSlicer as slice
import TileClassifier as hist
import MatrixCreator as matrix
import CrownFinding as crownFinder
import KingDominoScoreProgram as main

imageDir = "GroundTruth//BrickTruthImg"
testBoards = []

if os.path.isdir(imageDir):
    for file in os.listdir(imageDir):
        full_path = os.path.join(imageDir, file)
        img = cv2.imread(full_path)
        if img is None:
            print(f"Could not load image: {full_path}")
        else:
            print(f"Image loaded successfully: {full_path}")
            testBoards.append(img)
else:
    print("This is not a funtional path!")


labels = ["forest", "grasslands", "wheat", "swamp", "mine", "lake", "unknown"]
tile_confusion_matrix = np.zeros((len(labels), len(labels)), int)

labels_crown = ["Detected", "Non detected"]
crown_confusion_matrix = np.zeros((len(labels_crown), len(labels_crown)), int)

tile_index = {
    "forest": 0,
    "grassplane": 1,
    "wheat": 2,
    "swamp": 3,
    "mine": 4,
    "lake": 5,
    "unknown": 6
}


tileDict = {"TileType": np.uint8(0),
             "Crowns": np.uint8(0),
             "checked": False,
             "ImageID": np.uint8(0)}

groundtruth_tiles = []
groundtruth_crowns = []
groundtruth_score = []
allCroppedImages = []

with open("GroundTruth//BrickGroundTruth.txt") as f:
  for x in f:
    groundtruth_tiles.append(f"{x.replace(f"\n", "")}")

with open("GroundTruth//CrownGroundTruth.txt") as f:
    for x in f:
      groundtruth_crowns.append(int(x))

with open("GroundTruth//ScoreGroundTruth.txt") as f:
    for x in f:
        groundtruth_score.append(int(x))

def evaluate(programMatrixes):
    programResults = []
    confusMat = tile_confusion_matrix.copy()
    confusMatCrowns = crown_confusion_matrix.copy()
    var = 0
    for mat in programMatrixes:
        for i, row in enumerate(mat):
            for j, img in enumerate(row):
                programResults.append(mat[i, j]["TileType"])
                crownsFound = crownFinder.find_crowns(allCroppedImages[var])
                if crownsFound < groundtruth_crowns[var]:
                    confusMatCrowns[0, 1] += (groundtruth_crowns[var] - crownsFound)
                    confusMatCrowns[0, 0] += crownsFound
                elif crownsFound > groundtruth_crowns[var]:
                    confusMatCrowns[1, 0] += (crownsFound - groundtruth_crowns[var])
                    confusMatCrowns[0, 0] += (crownsFound - (crownsFound - groundtruth_crowns[var]))
                elif crownsFound == groundtruth_crowns[var]:
                    if crownsFound != 0:
                        confusMatCrowns[0, 0] += crownsFound

                var += 1
    print(len(programResults))
    for x, truth in enumerate(groundtruth_tiles):
        confusMat[tile_index[programResults[x]], tile_index[truth]] += 1


    return confusMat, confusMatCrowns

matrixes = []


overall_error_rate = 0

for i, score in enumerate(groundtruth_score):
    programScore = main.KingDominoScore(testBoards[i])
    variance = math.pow((score - programScore), 2)
    errorRate = math.sqrt(variance) / score
    overall_error_rate += errorRate
    print(f"Program score: {programScore} ; Real Score: {score}")
    print(f"Board {i} have an error rate of {errorRate}")

average_error_rate = overall_error_rate/len(groundtruth_score)

print(f"Average Error Rate is {average_error_rate}")

crowns = 0
for boards in testBoards:
    blur = cv2.blur(boards, (23, 23))
    sharpened_img_org = cv2.addWeighted(boards, 1.7, blur, -0.8, 0)

    croppedImages = slice.slice_image(sharpened_img_org)

    tileTypes = []
    nrOfCrowns = []
    for croppedImage in croppedImages:
        allCroppedImages.append(croppedImage)
        tileTypes.append(hist.classify_tile(croppedImage))
        nrOfCrowns.append(crownFinder.find_crowns(croppedImage))
        crowns += crownFinder.find_crowns(croppedImage)

    matrixes.append(matrix.createMatrix(tileTypes, nrOfCrowns))


confTile, confCrown = evaluate(matrixes)


print(confTile)
print(confCrown)

print(crowns)

totalCrowns = 0

for crowns in groundtruth_crowns:
    totalCrowns += crowns

print(totalCrowns)
