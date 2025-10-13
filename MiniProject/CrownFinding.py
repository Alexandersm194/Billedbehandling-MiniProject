import cv2
import numpy as np

from matplotlib import pyplot as plt

#load image
#img = cv2.imread("Templates/CrownTemp3.jpg")
crownTemps = [
    cv2.imread("Templates//CrownTemp.jpg", cv2.IMREAD_GRAYSCALE),
    cv2.imread("Templates//CrownTemp1.jpg", cv2.IMREAD_GRAYSCALE),
    cv2.imread("Templates//CrownTemp2.jpg", cv2.IMREAD_GRAYSCALE),
    cv2.imread("Templates//CrownTemp3.jpg", cv2.IMREAD_GRAYSCALE),
]

crownTempsEdges = []

for crown in crownTemps:
    crownTempsEdges.append(cv2.Canny(crown, 195, 200))

def crownEdges(img):
    nrOfCrowns = 0
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canny_edges = cv2.Canny(gray_image, 195, 200)

    for crownTemp in crownTempsEdges:
        matchTemp = cv2.matchTemplate(canny_edges, crownTemp, cv2.TM_CCOEFF_NORMED)
        thres = 0.4
        _, threshold = cv2.threshold(matchTemp, thres, 1, cv2.THRESH_BINARY)
        finalImage = (threshold * 255).astype(np.uint8)
        contours, _ = cv2.findContours(finalImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        nrOfCrowns += len(contours)
    cv2.imshow("Crown Edges", canny_edges)
    cv2.imshow("Crown Temps", crownTempsEdges[0])
    cv2.waitKey(0)
    return nrOfCrowns


'''
#calculating moments of image
M = cv2.moments(canny_edges)
#calculating x y coordinates of the center
cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])

# show result
fig = plt.figure(figsize = (100,100))
plt.imshow(canny_edges)
plt.xticks([])
plt.yticks([])
plt.show()
print(cx, cy)
#cv2.imshow("Canny Edges", gray_image)
#cv2.waitKey(0)'''