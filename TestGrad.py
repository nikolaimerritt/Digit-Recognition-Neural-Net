import FourLayerNet
import numpy as np
from mnist import MNIST


def outLayerFromLabel(label: int):
    outputLayer = []
    for i in range(10):
        outputLayer.append(0 if i != label else 1)
    return outputLayer


mndata = MNIST("samples")
images, labels = mndata.load_training()
desOutLayers = [outLayerFromLabel(label) for label in labels]

inLayer = images[0]
desOutLayer = desOutLayers[0]
params = FourLayerNet.getRandomParams(len(images[0]), 16, 16, 10)


def basisBias(biasNum, biasIdx):
    basis = 0 * params
    basis.biases[biasNum][biasIdx] = 1
    return basis


def basisWeight(weightNum, weightIdx, weightCol):
    basis = 0 * params
    basis.weights[weightNum][weightIdx][weightCol] = 1
    return basis


def numericBiasDeriv(biasNum, biasIdx):
    basisParam = basisBias(biasNum, biasIdx)
    change = lambda h: FourLayerNet.singleCost(inLayer, desOutLayer, params + h * basisParam)
    epsilon = 10 ** (-5)
    return (change(epsilon) - change(-epsilon)) / (2 * epsilon)


def testBias(biasNum, biasIdx):
    gradToTest = FourLayerNet.paramsGrad(inLayer, desOutLayer, params)
    componentToTest = gradToTest.biases[biasNum][biasIdx]
    numericComponent = numericBiasDeriv(biasNum, biasIdx)

    #print("component to test", componentToTest)
    #print("component found numerically", numericComponent)
    return abs(componentToTest - numericComponent) / numericComponent


def testAllBiases():
    for biasNum in [2, 1, 0]:
        for biasIdx in range(len(params.biases[biasNum])):
            relError = testBias(biasNum, biasIdx)
            if relError > 10 ** (-5):
                print("relative error", relError)
                print("biasNum", biasNum)
                print("biasIdx", biasIdx)
    
testAllBiases()