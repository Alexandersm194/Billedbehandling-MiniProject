import cv2
import numpy as np


def slice_image(orgImage):
    kernel_size_x = orgImage.shape[0] // 5
    kernel_size_y = orgImage.shape[1] // 5
    croppedImages = []
    for y in range(5):
        for x in range(5):
            if(y == 0 and x == 0):
                croppedImages.append(orgImage[0:kernel_size_y, 0:kernel_size_x])
            elif(y == 0 and x != 0):
                croppedImages.append(orgImage[0:kernel_size_y, kernel_size_x * x:kernel_size_x * x + kernel_size_x])
            elif(y != 0 and x == 0):
                croppedImages.append(orgImage[kernel_size_y * y:kernel_size_y * y + kernel_size_y, 0:kernel_size_x])
            elif(y != 0 and x != 0):
                croppedImages.append(orgImage[kernel_size_y * y:kernel_size_y * y + kernel_size_y, kernel_size_x * x:kernel_size_x * x + kernel_size_x])

    return croppedImages