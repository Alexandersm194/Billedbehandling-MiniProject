import CrownFinding as cf
import cv2
import ImageSlicer as slice
import TileClassifier as tc
import MatrixCreator as matrix
import ScoreCalculator as sc

def KingDominoScore(image):
    tileTypes = []
    numberOfCrowns = []
    blur = cv2.blur(image, (23, 23))
    sharpened_img_org = cv2.addWeighted(image, 1.7, blur, -0.8, 0)


    croppedImages = slice.slice_image(sharpened_img_org)

    for croppedImage in croppedImages:
        tileTypes.append(tc.classify_tile(croppedImage))
        numberOfCrowns.append(cf.find_crowns(croppedImage))

    full_board_matrix = matrix.createMatrix(tileTypes, numberOfCrowns)

    final_score = sc.counter(full_board_matrix)

    return final_score

final_score_out = KingDominoScore(cv2.imread("GroundTruth//BrickTruthImg//65.jpg"))
print(final_score_out)
