import cv2


image = cv2.imread("King Domino dataset//Cropped and perspective corrected boards//9.jpg")

kernel_size_x = image.shape[0] // 5
kernel_size_y = image.shape[1] // 5

kernel_radius_x = kernel_size_x // 2
kernel_radius_y = kernel_size_y // 2

croppedImages = []

for y in range(5):
    for x in range(5):
        pixelValues = [0, 0]
        if(y == 0 and x == 0):
            pixelValues = [kernel_radius_y, kernel_radius_x]
            print(pixelValues)
            croppedImages.append(image[0:kernel_size_y, 0:kernel_size_x])
        elif(y == 0 and x != 0):
            pixelValues = [kernel_radius_y, kernel_radius_x + kernel_size_x * x]
            croppedImages.append(image[0:kernel_size_y, kernel_size_x * x:kernel_size_x * x + kernel_size_x])
            print(pixelValues)
        elif(y != 0 and x == 0):
            pixelValues = [kernel_radius_y + kernel_size_y * y, kernel_radius_x]
            croppedImages.append(image[kernel_size_y * y:kernel_size_y * y + kernel_size_y, 0:kernel_size_x])
        elif(y != 0 and x != 0):
            pixelValues = [kernel_radius_y + kernel_size_y * y, kernel_radius_x + kernel_size_x * x]
            croppedImages.append(image[kernel_size_y * y:kernel_size_y * y + kernel_size_y, kernel_size_x * x:kernel_size_x * x + kernel_size_x])

            print(pixelValues)



print(len(croppedImages))

i = 1
for croppedImage in croppedImages:
    cv2.imshow(f"square {i}", croppedImage)
    i = i + 1
    cv2.waitKey(0)

cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()