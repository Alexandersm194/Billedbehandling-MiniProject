import cv2
import ImageSlicer
import CrownFinding
import TileClassifier
import matplotlib.pyplot as plt


image = cv2.imread("King Domino dataset/FullBoardsTestData/70.jpg")
cv2.imshow("image", image)
slicedImages = ImageSlicer.slice_image(image)

Crowns = CrownFinding.find_crowns(slicedImages[5])

'''hist1 = TileClassifier.calculate_hue_hist(slicedImages[14])
hist2 = TileClassifier.calculate_hue_hist(slicedImages[24])

plt.subplot(121), plt.plot(hist1)
plt.subplot(122), plt.plot(hist2)
plt.show()'''

cv2.waitKey(0)
cv2.destroyAllWindows()