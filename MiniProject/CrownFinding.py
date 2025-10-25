import cv2
import numpy as np

crownTemps = [
    cv2.imread("Templates//CrownTemp.jpg"),
    cv2.imread("Templates//CrownTemp1.jpg"),
    cv2.imread("Templates//CrownTemp2.jpg"),
    cv2.imread("Templates//CrownTemp3.jpg")
]

def sharpen(orgimg):
    blur = cv2.blur(orgimg, (23, 23))
    median = cv2.medianBlur(orgimg, 23)

    gaussian_blur = cv2.GaussianBlur(orgimg, (23, 23), 0)
    # adding the two pictures together with larger weight on the original image :D
    return cv2.addWeighted(orgimg, 1.7, blur, -0.8, 0)

crownTempsEdges = []

for crown in crownTemps:
    sharpenImg = sharpen(crown)
    edge = cv2.Canny(crown, 195, 200)
    crownTempsEdges.append(edge)

def find_crowns(img):
    nrOfCrowns = 0
    canny_edges = cv2.Canny(img, 195, 200)
    cv2.imshow("Orginal", img)
    cv2.imshow("Canny edges", canny_edges)
    cv2.waitKey(0)

    for crownTemp in crownTempsEdges:
        matchTemp = cv2.matchTemplate(canny_edges, crownTemp, cv2.TM_CCOEFF_NORMED)
        thres = 0.22
        #thres = 0.445
        _, threshold = cv2.threshold(matchTemp, thres, 1, cv2.THRESH_BINARY)
        cv2.imshow("match", matchTemp)
        cv2.imshow(f"thres", threshold)
        cv2.waitKey(0)
        finalImage = (threshold * 255).astype(np.uint8)
        contours, _ = cv2.findContours(finalImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        nrOfCrowns += len(contours)

    return nrOfCrowns
