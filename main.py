import numpy as np
from mnist import MNIST
import FourLayerNet
from Params import loadFromFile

def outLayerFromLabel(label: int):
    return np.array([1 if i == label else 0 for i in range(10)])


def mnistDataToNumpyArrays(images, labels):
    inLayers = [np.array(img) for img in images]
    desOutLayers = [outLayerFromLabel(label) for label in labels]
    return inLayers, desOutLayers

mndata = MNIST("samples")
trainingInLayers, trainingDesOutLayers = mnistDataToNumpyArrays(*mndata.load_training())
testingInLayers, testingDesOutLayers = mnistDataToNumpyArrays(*mndata.load_testing())


params = loadFromFile()
print(f"{100 * FourLayerNet.proportionCorrect(trainingInLayers, trainingDesOutLayers, params):.2f}% of the training data guessed correctly")
print(f"{100 * FourLayerNet.proportionCorrect(testingInLayers, testingDesOutLayers, params):.2f}% of the testing data guessed correctly")