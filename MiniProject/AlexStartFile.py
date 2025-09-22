import cv2
import numpy as np

inputImage = cv2.imread("King Domino dataset//Cropped and perspective corrected boards//12.jpg", cv2.IMREAD_GRAYSCALE)

crownTemp = cv2.imread("Templates//CrownTemp.jpg", cv2.IMREAD_GRAYSCALE)
crownTemp1 = cv2.imread("Templates//CrownTemp1.jpg", cv2.IMREAD_GRAYSCALE)
crownTemp2 = cv2.imread("Templates//CrownTemp2.jpg", cv2.IMREAD_GRAYSCALE)
crownTemp3 = cv2.imread("Templates//CrownTemp3.jpg", cv2.IMREAD_GRAYSCALE)

matchTemp = cv2.matchTemplate(inputImage, crownTemp, cv2.TM_CCOEFF_NORMED)
matchTemp1 = cv2.matchTemplate(inputImage, crownTemp1, cv2.TM_CCOEFF_NORMED)
matchTemp2 = cv2.matchTemplate(inputImage, crownTemp2, cv2.TM_CCOEFF_NORMED)
matchTemp3 = cv2.matchTemplate(inputImage, crownTemp3, cv2.TM_CCOEFF_NORMED)

thres = 0

_, threshold = cv2.threshold(matchTemp, thres, 1, cv2.THRESH_BINARY)
_, threshold1 = cv2.threshold(matchTemp1, thres, 1, cv2.THRESH_BINARY)
_, threshold2 = cv2.threshold(matchTemp2, thres, 1, cv2.THRESH_BINARY)
_, threshold3 = cv2.threshold(matchTemp3, thres, 1, cv2.THRESH_BINARY)

addImg1 = cv2.addWeighted(threshold, 0.5, threshold2, 0.5, 0)
addImg2 = cv2.addWeighted(threshold1, 0.5, threshold3, 0.5, 0)

addImg1 = (addImg1 * 255).astype(np.uint8)
addImg2 = (addImg2 * 255).astype(np.uint8)

contours1, _ = cv2.findContours(addImg1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contours2, _ = cv2.findContours(addImg1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

h1, w1 = matchTemp.shape[:2]
h2, w2 = matchTemp1.shape[:2]
centers = []

print(f"{len(contours1)} and {len(contours2)}")

for cnt in contours1:
    x, y, bw, bh = cv2.boundingRect(cnt)
    center = (x + bw // 2 + w1 // 2, y + bh // 2 + h1 // 2)
    centers.append(center)

for cnt in contours2:
    x, y, bw, bh = cv2.boundingRect(cnt)
    center = (x + bw // 2 + w2 // 2, y + bh // 2 + h2 // 2)
    centers.append(center)


print("Centers found:", len(centers))

cv2.imshow("Image", addImg1)
cv2.imshow("Image 2", addImg2)

cv2.waitKey(0)

cv2.destroyAllWindows()