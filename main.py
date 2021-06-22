import numpy as np
from mnist import MNIST
import FourLayerNet
from Params import loadFromFile

def outLayerFromLabel(label: int):
    return np.array([1 if i == label else 0 for i in range(10)])

mndata = MNIST("samples")
images, labels = mndata.load_training()
inputLayers = [np.array(img) for img in images]
outputLayers = [outLayerFromLabel(l) for l in labels]

params = FourLayerNet.getRandomParams(len(inputLayers[0]), 16, 16, 10)
params = FourLayerNet.batchGradDescent(inputLayers, outputLayers, params)
params.saveToFile()
guess = FourLayerNet.calcOutLayer(inputLayers[0], params)
print(guess)
print(labels[0])