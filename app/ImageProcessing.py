from PIL import Image
import numpy as np

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


def inLayerToString(inLayer: np.ndarray, width=28):
    string = ""
    for rowIdx in range(0, len(inLayer), width):
        row = inLayer[rowIdx : rowIdx + width]
        string += "\t".join([str(int(x)) for x in row]) 
        string += "\n"
    return string