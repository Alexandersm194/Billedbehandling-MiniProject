import numpy as np

var = ""
def createMatrix(brickDict, brickTypes, areaBrickDict):
    matrix = np.empty((5, 5), dtype=object)
    for i in range(5):
        for j in range(5):
            matrix[i, j] = brickDict.copy()

    croppedImgIndex = 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):

            if brickTypes[croppedImgIndex] is None:
                var = ""
            else:
                var = brickTypes[croppedImgIndex]

            matrix[i, j]["BrickType"] = var
            matrix[i, j]["ImageID"] = croppedImgIndex
            croppedImgIndex += 1

    return matrix

