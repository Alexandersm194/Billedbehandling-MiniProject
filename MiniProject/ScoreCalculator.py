import numpy as np

propertyDict = {"TileType": "",
                "Count": np.uint8(0),
                "Crowns": np.uint8(0)}
def dfs(matrix, x, y, TileType, in_count, crowns):
    if (x < 0 or x >= 5 or y < 0 or y >= 5 or matrix[y, x]["checked"] == True
            or matrix[y, x]["TileType"] != TileType):
        return in_count, crowns

    matrix[y, x]["checked"] = True

    in_count = in_count + 1

    crowns = crowns + matrix[y, x]["Crowns"]

    in_count, crowns = dfs(matrix, x + 1, y, TileType, in_count, crowns)
    in_count, crowns = dfs(matrix, x - 1, y, TileType, in_count, crowns)
    in_count, crowns = dfs(matrix, x, y + 1, TileType, in_count, crowns)
    in_count, crowns = dfs(matrix, x, y - 1, TileType, in_count, crowns)

    return in_count, crowns

def calculate_final_score(properties):
    final_score = 0
    for prop in properties:
        final_score += prop["Count"] * prop["Crowns"]

    return final_score

properties = []
def counter(matrix):
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            if not matrix[y, x]["checked"]:
                tileType = matrix[y, x]["TileType"]
                count, crowns = dfs(matrix, x, y, tileType, 0, 0)
                matrix[y, x]["checked"] = True

                prop = propertyDict.copy()
                prop["TileType"] = tileType
                prop["Count"] = count
                prop["Crowns"] = crowns
                properties.append(prop)

    print(properties)
    return calculate_final_score(properties)

