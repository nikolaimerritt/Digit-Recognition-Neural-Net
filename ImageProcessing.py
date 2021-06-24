from typing import ForwardRef
from PIL import Image, ImageDraw
import math
import Params, FourLayerNet
import numpy as np
import tkinter
from mnist import MNIST
import mnist

def pixelate(image: Image, pixels: int):
    reducedImg = image.resize((pixels, pixels), Image.NEAREST)
    magnifiedImg = reducedImg.resize((reducedImg.size[0] * pixels, reducedImg.size[1] * pixels), Image.NEAREST)
    return reducedImg, magnifiedImg


def imageToInputLayer(image: Image, pixels: int, rgbIx=0):
    inputLayer = [
        image.getpixel((r, c))[rgbIx]
        for c in range(pixels) for r in range(pixels)
    ]
    return np.array(inputLayer)
    
    


appSize = 400
pixels = 28
lineWidth = math.ceil(appSize / pixels)
app = tkinter.Tk()

canvas = tkinter.Canvas(app, width=appSize, height=appSize, bg="black")
canvas.pack()

image = Image.new("RGB", (appSize, appSize), (0, 0, 0))
drawing = ImageDraw.Draw(image)

def getXandY(event):
    global lastX, lastY
    lastX, lastY = event.x, event.y


def drawLine(event):
    global lastX, lastY
    canvas.create_line((lastX, lastY, event.x, event.y), fill="red", width=lineWidth)
    drawing.line([(lastX, lastY), (event.x, event.y)], fill="red", width=lineWidth)
    lastX, lastY = event.x, event.y


def onWindowClose():
    global image
    reducedImg, magnifiedImg = pixelate(image, pixels)
    magnifiedImg.show()
    inLayer = imageToInputLayer(reducedImg, pixels)
    print(f"Digit recognised as {FourLayerNet.recogniseDigit(inLayer, Params.loadFromFile())}")
    
    app.destroy()


canvas.bind("<Button-1>", getXandY)
canvas.bind("<B1-Motion>", drawLine)

app.protocol("WM_DELETE_WINDOW", onWindowClose)
app.mainloop()
