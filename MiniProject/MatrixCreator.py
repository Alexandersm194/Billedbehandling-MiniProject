import numpy as np

tileDict = {"TileType": str,
             "Crowns": np.uint8(0),
             "checked": False,
             "ImageID": np.uint8(0)}
def createMatrix(tileTypes, nrOfCrowns):
    matrix = np.empty((5, 5), dtype=object)
    for i in range(5):
        for j in range(5):
            matrix[i, j] = tileDict.copy()

    croppedImgIndex = 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):

            if tileTypes[croppedImgIndex] is None:
                tile_var = ""
            else:
                tile_var = tileTypes[croppedImgIndex]
                crown_var = nrOfCrowns[croppedImgIndex]

            matrix[i, j]["TileType"] = tile_var
            matrix[i, j]["Crowns"] = crown_var
            matrix[i, j]["ImageID"] = croppedImgIndex
            croppedImgIndex += 1

    return matrix
