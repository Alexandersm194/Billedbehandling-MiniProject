import random
import CrownFinding as crownFinder
import cv2
import os
import numpy as np
import ImageSlicer as slice
import HistComparison as hist
import MatrixCreator as matrix
import CrownFinding as crownFinder

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
brick_confusion_matrix = np.zeros((len(labels), len(labels)), dtype=int)

labels_crown = ["Detected", "Non detected"]
crown_confusion_matrix = np.zeros((len(labels_crown), len(labels_crown)), dtype=int)

brick_index = {
    "forest": 0,
    "grassplane": 1,
    "wheat": 2,
    "swamp": 3,
    "mine": 4,
    "lake": 5,
    "unknown": 6
}

'''areaBrickDict = {"forest": 0,
                  "swamp": 1,
                  "mine": 2,
                  "grassplane": 3,
                  "lake": 4,
                  "wheat": 5,
                 "unknown": 6}'''
areaBrickDict = ["forest", "grassplane", "lake", "mine", "swamp", "wheat", "unknown"]

areaBrickIndex = ["forest", "grasslands", "wheat", "mine", "lake", "unknown"]

brickDict = {"BrickType": np.uint8(0),
             "Crowns": np.uint8(0),
             "checked": False,
             "ImageID": np.uint8(0)}

groundtruth = []
groundtruth_crowns = []
allCroppedImages = []

with open("GroundTruth//BrickGroundTruth.txt") as f:
  for x in f:
    groundtruth.append(f"{x.replace(f"\n", "")}")

with open("GroundTruth//CrownGroundTruth.txt") as f:
    for x in f:
      groundtruth_crowns.append(int(x))

def system_precision_recall(confMat):
    confMat = np.array(confMat)

    if isinstance(confMat[0][0], str):
        confMat = confMat[1:, 1:].astype(int)

    TP = np.trace(confMat)
    total_predicted = np.sum(confMat)
    FP = total_predicted - TP
    FN = FP

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    return precision, recall
def evaluate(programMatrixes):
    programResults = []
    confusMat = brick_confusion_matrix.copy()
    confusMatCrowns = crown_confusion_matrix.copy()
    var = 0
    for mat in programMatrixes:
        for i, row in enumerate(mat):
            for j, img in enumerate(row):
                programResults.append(mat[i, j]["BrickType"])
                crownsFound = crownFinder.crownEdges(allCroppedImages[var])
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

    for x, truth in enumerate(groundtruth):
        confusMat[brick_index[programResults[x]], brick_index[truth]] += 1


    return confusMat, confusMatCrowns

matrixes = []

crowns = 0
for boards in testBoards:
    croppedImages = slice.slice_image(boards)

    brickTypes = []
    for croppedImage in croppedImages:
        allCroppedImages.append(croppedImage)
        brickTypes.append(hist.classify_brick(croppedImage))
        crowns += crownFinder.crownEdges(croppedImage)

    matrixes.append(matrix.createMatrix(brickDict, brickTypes, areaBrickDict))


confBrick, confCrown = evaluate(matrixes)


precision, recall = system_precision_recall(confBrick)
print(confBrick)
print(confCrown)
print(precision)
print(recall)
print(crowns)

totalCrowns = 0

for crowns in groundtruth_crowns:
    totalCrowns += crowns

print(totalCrowns)
