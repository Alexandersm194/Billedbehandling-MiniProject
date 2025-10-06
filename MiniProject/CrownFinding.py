import cv2

img = cv2.imread("King Domino dataset//Cropped and perspective corrected boards//test.png")
crownTemp = cv2.imread("Templates//CrownTemp.jpg")

rotated = cv2.rotate(crownTemp, cv2.ROTATE_180)

cv2.imshow("180 Clockwise", rotated)
cv2.waitKey(0)
cv2.destroyAllWindows()



'''img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cnt = contours[4]
cv2.drawContours(img, [cnt], 0, (0, 255, 0), 3)

cv2.imshow("InputImage", img)
cv2.waitKey(0)
cv2.destroyAllWindows()'''
