import cv2
import numpy as np

crownTemps = [
    cv2.imread("Templates//CrownTemp.jpg"),
    cv2.imread("Templates//CrownTemp1.jpg"),
    cv2.imread("Templates//CrownTemp2.jpg"),
    cv2.imread("Templates//CrownTemp3.jpg")]

crownTempsEdges = []

for crown in crownTemps:
    crownTempsEdges.append(cv2.Canny(crown, 195, 200))

def find_crowns(img):
    nrOfCrowns = 0
    canny_edges = cv2.Canny(img, 195, 200)

    for crownTemp in crownTempsEdges:
        matchTemp = cv2.matchTemplate(canny_edges, crownTemp, cv2.TM_CCOEFF_NORMED)
        thres = 0.22
        _, threshold = cv2.threshold(matchTemp, thres, 1, cv2.THRESH_BINARY)
        finalImage = (threshold * 255).astype(np.uint8)
        contours, _ = cv2.findContours(finalImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        nrOfCrowns += len(contours)

    return nrOfCrowns
