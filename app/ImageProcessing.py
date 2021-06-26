from PIL import Image
from typing import List
import numpy as np
import math

# TODO: centre the images once they are drawn
# training and testing data is centred
# and images drawn centred are predicted way better
# " the error rate improves when the digits are centered by bounding box" -- MNIST database

def inputLayerToMatrix(inLayer: np.ndarray, width=28) -> np.ndarray:
    return np.ndarray([
        inLayer[startIdx : startIdx + width] 
        for startIdx in range(0, len(inLayer), width)
    ])


def gridToInLayer(grid: List[np.ndarray]):
    return np.ndarray([
        grid[r][c] 
        for c in range(len(grid[0]))
        for r in range(len(grid))
    ])


def cropGridVertically(grid: np.ndarray):
    # cropping top bit off
    topRow = 0
    while topRow < len(grid) and all(grid[topRow][c] == 0 for c in range(len(grid[topRow]))):
        topRow += 1
    
    # cropping bottom bit off
    bottomRow = len(grid) - 1
    while bottomRow >= 0 and all(grid[bottomRow][c] == 0 for c in range(len(grid[bottomRow]))):
        bottomRow -= 1
    
    return grid[topRow : bottomRow + 1]


def cropGrid(grid: np.ndarray):
    vertCropped = cropGridVertically(grid)
    horizCropped = cropGridVertically(vertCropped.T).T
    return horizCropped


def padBoundedBox(grid: np.ndarray, width=28, height=28) -> np.ndarray:
    widthDelta = (width - len(grid[0])) / 2
    leftPadding = math.floor(widthDelta)
    rightPadding = math.ceil(widthDelta)
    
    heightDelta = (height - len(grid)) / 2
    topPadding = math.ceil(heightDelta)
    bottomPadding = math.floor(heightDelta)

    return np.pad(grid, ((topPadding, bottomPadding), (leftPadding, rightPadding)))
    

def pixelate(image: Image, pixels: int):
    reducedImg = image.resize((pixels, pixels), Image.LINEAR)
    magnifiedImg = reducedImg.resize((reducedImg.size[0] * pixels, reducedImg.size[1] * pixels), Image.NEAREST)
    return reducedImg, magnifiedImg


def imageToInputLayer(image: Image, pixels: int, rgbIx=0):
    inputLayer = [
        image.getpixel((r, c))[rgbIx]
        for c in range(pixels) for r in range(pixels)
    ]
    return np.array(inputLayer)


def centreInputLayer(inLayer: np.ndarray, width=28, height=28):
    grid = inLayer.reshape((width, height))
    grid = padBoundedBox(cropGrid(grid))
    return grid.reshape(width * height)


def inLayerToString(inLayer: np.ndarray, width=28):
    string = ""
    for rowIdx in range(0, len(inLayer), width):
        row = inLayer[rowIdx : rowIdx + width]
        string += "\t".join([str(int(x)) for x in row]) 
        string += "\n"
    return string


def displayOutLayer(outLayer: np.ndarray, width=20) -> str:
    string = ""
    for i in range(len(outLayer)):
        percentChance = 100 * outLayer[i]
        lengthOfBar = math.ceil(width * outLayer[i])
        bar = "#" * lengthOfBar + "|" + " " * (width - lengthOfBar)
        string += f"{i}: " + bar + f"\t{str(percentChance)[:4]} %" + "\n"
    return string