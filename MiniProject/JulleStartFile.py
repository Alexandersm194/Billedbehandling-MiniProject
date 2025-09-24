import cv2
import matplotlib.pyplot as plt
from cv2 import waitKey, destroyAllWindows
from matplotlib.testing.compare import crop_to_same

img = cv2.imread("King Domino dataset/Cropped and perspective corrected boards/1.jpg")

kernelSizeX = img.shape[0] // 5
kernelSizeY = img.shape[1] // 5





'''cv2.imshow("Original", img)'''
plt.imshow(img)
plt.show()
print(img.shape)

waitKey(0)
destroyAllWindows()