from typing import List
import numpy as np
from mnist import MNIST
import FourLayerNet, Params

def outLayerFromLabel(label: int):
    return [1 if i == label else 0 for i in range(10)]


def filterImage(image: List[int]):
    """ filters images by the same filter as mnist.display does """
    return [0 if entry <= 200 else 255 for entry in image]


def mnistDataToNumpyArrays(images, labels):
    inLayers = [np.array(filterImage(img)) for img in images]
    desOutLayers = [np.array(outLayerFromLabel(label)) for label in labels]
    return inLayers, desOutLayers


mndata = MNIST("samples")
images, _ = mndata.load_training()

trainingIn, trainingDesOut = mnistDataToNumpyArrays(*mndata.load_training())
# testingInLayers, testingDesOutLayers = mnistDataToNumpyArrays(*mndata.load_testing())


# params = FourLayerNet.getRandomParams(len(trainingIn[0]), 16, 16, 10)
params = FourLayerNet.batchGradDescent(trainingIn, trainingDesOut, Params.loadFromFile())
# print(f"{100 * FourLayerNet.proportionCorrect(trainingInLayers, trainingDesOutLayers, params):.2f}% of the training data guessed correctly")
# print(f"{100 * FourLayerNet.proportionCorrect(testingInLayers, testingDesOutLayers, params):.2f}% of the testing data guessed correctly")
