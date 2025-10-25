import cv2
import ImageSlicer
import CrownFinding
import TileClassifier
import matplotlib.pyplot as plt


image = cv2.imread("King Domino dataset/FullBoardsTestData/65.jpg")
blur = cv2.blur(image, (23, 23))
sharpened_img_org = cv2.addWeighted(image, 1.7, blur, -0.8, 0)
cv2.imshow("image", sharpened_img_org)
slicedImages = ImageSlicer.slice_image(sharpened_img_org)


hist1 = TileClassifier.calculate_hue_hist(slicedImages[1])
hist2 = TileClassifier.calculate_hue_hist(slicedImages[5])
hist3 = TileClassifier.calculate_hue_hist(slicedImages[10])

plt.plot(hist1, label="Grassland", color='r')
plt.plot(hist2, label="Forest", color='lightgreen')
plt.plot(hist3, label="Crown", color='g')

plt.xlabel('Hue')
plt.ylabel('Frequency')

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()