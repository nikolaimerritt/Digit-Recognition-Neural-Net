from PIL import ImageDraw, Image
import tkinter
import math
from app.ImageProcessing import pixelate, imageToInputLayer, inLayerToString
from NeuralNet import Params, FourLayerNet, util

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
    canvas.create_line((lastX, lastY, event.x, event.y), fill="red", width=lineWidth)
    drawing.line([(lastX, lastY), (event.x, event.y)], fill="red", width=lineWidth)
    lastX, lastY = event.x, event.y


def onWindowClose():
    global image
    reducedImg, magnifiedImg = pixelate(image, pixels)
    inLayer = imageToInputLayer(reducedImg, pixels)
    with open("images/my-layer.txt", "w") as f:
        f.write(inLayerToString(inLayer))
    
    params = Params.loadFromFile()
    print(f"Digit recognised as \n{util.displayProbVector(FourLayerNet.calcOutLayer(inLayer, params))}")
    
    app.destroy()


canvas.bind("<Button-1>", getXandY)
canvas.bind("<B1-Motion>", drawLine)

app.protocol("WM_DELETE_WINDOW", onWindowClose)

def launch():
    app.mainloop()