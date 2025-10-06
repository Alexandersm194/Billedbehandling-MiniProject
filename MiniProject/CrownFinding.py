import cv2

from matplotlib import pyplot as plt

#load image
#img = cv2.imread("Templates/CrownTemp3.jpg")

def crownEdges(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canny_edges = cv2.Canny(gray_image, 195, 200)
    return canny_edges

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