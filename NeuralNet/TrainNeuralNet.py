import numpy as np
from mnist import MNIST
from NeuralNet import FourLayerNet
from NeuralNet.Params import Params

def outLayerFromLabel(label: int):
    return [1 if i == label else 0 for i in range(10)]


def mnistDataToNumpyArrays(images, labels):
    inLayers = [np.array(img) for img in images]
    desOutLayers = [np.array(outLayerFromLabel(label)) for label in labels]
    return inLayers, desOutLayers


def train():
    mndata = MNIST("training-and-test-data")

    trainInLayers, trainingDesOutLayers = mnistDataToNumpyArrays(*mndata.load_training())
    testInLayers, testDesOutLayers = mnistDataToNumpyArrays(*mndata.load_testing())
    
    params = Params.loadFromFile()
    #params = Params.random([len(trainInLayers[0]), 200, 80, len(trainingDesOutLayers[0])])
    params = FourLayerNet.saveBatchGradDescent(trainInLayers, trainingDesOutLayers, params, testInLayers, testDesOutLayers)

    print(f"{100 * FourLayerNet.proportionCorrect(trainInLayers, trainingDesOutLayers, params):.2f}% of the training data guessed correctly")
    print(f"{100 * FourLayerNet.proportionCorrect(testInLayers, testDesOutLayers, params):.2f}% of the testing data guessed correctly")