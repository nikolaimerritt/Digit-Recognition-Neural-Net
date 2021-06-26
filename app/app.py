from tkinter.constants import W
from PIL import ImageDraw, Image
import tkinter
import math
from mnist import MNIST
from app import ImageProcessing
from app.ImageProcessing import pixelate, imageToInputLayer, inLayerToString
from NeuralNet import FourLayerNet
from NeuralNet.Params import Params

appSize = 400
pixels = 28
lineWidth = 2 * math.ceil(appSize / pixels)

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
    
    canvas.create_line((lastX, lastY, event.x, event.y), fill="red", width=lineWidth, capstyle=tkinter.ROUND)
    drawing.line([(lastX, lastY), (event.x, event.y)], fill="red", width=lineWidth, joint="curve")

    lastX, lastY = event.x, event.y


def onWindowClose():
    global image
    reducedImg, magnifiedImg = pixelate(image, pixels)
    
    inLayer = imageToInputLayer(reducedImg, pixels)
    centredInLayer = ImageProcessing.centreInputLayer(inLayer)
    
    params = Params.loadFromFile("params-values")
    outLayer = FourLayerNet.calcOutLayer(centredInLayer, params)
    print(ImageProcessing.displayOutLayer(outLayer, width=50))
    
    app.destroy()


canvas.bind("<Button-1>", getXandY)
canvas.bind("<B1-Motion>", drawLine)

app.protocol("WM_DELETE_WINDOW", onWindowClose)

def launch():
    app.mainloop()