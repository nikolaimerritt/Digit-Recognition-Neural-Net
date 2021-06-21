import FourLayerNet
import numpy as np
from mnist import MNIST


def outLayerFromLabel(label: int):
    return np.array([1 if i == label else 0 for i in range(10)])


mndata = MNIST("samples")
images, labels = mndata.load_training()
desOutLayers = [outLayerFromLabel(label) for label in labels]

inLayer = np.array(images[0])
desOutLayer = desOutLayers[0]
params = FourLayerNet.getRandomParams(len(images[0]), 16, 16, 10)
gradToTest = FourLayerNet.paramsGrad(inLayer, desOutLayer, params)
epsilon = 10 ** (-5)


def basisBias(biasNum, biasIdx):
    basis = 0 * params
    basis.biases[biasNum][biasIdx] = 1
    return basis


def basisWeight(weightNum, row, col):
    basis = 0 * params
    basis.weights[weightNum][row][col] = 1
    return basis


def numericBiasDeriv(biasNum, biasIdx):
    basisParam = basisBias(biasNum, biasIdx)
    change = lambda h: FourLayerNet.singleCost(inLayer, desOutLayer, params + h * basisParam)
    return (change(epsilon) - change(-epsilon)) / (2 * epsilon)


def relErrorInBias(biasNum, biasIdx):
    componentToTest = gradToTest.biases[biasNum][biasIdx]
    numericComponent = numericBiasDeriv(biasNum, biasIdx)
    error = abs(componentToTest - numericComponent)
    if numericComponent == 0:
        return error
    else:
        return error / numericComponent


def numericWeightDeriv(weightNum, row, col):
    basisParam = basisWeight(weightNum, row, col)
    change = lambda h: FourLayerNet.singleCost(inLayer, desOutLayer, params + h * basisParam)
    return (change(epsilon) - change(-epsilon)) / (2 * epsilon)
    

def relErrorInWeight(weightNum, row, col):
    componentToTest = gradToTest.weights[weightNum][row][col]
    numericComponent = numericWeightDeriv(weightNum, row, col)
    error = abs(componentToTest - numericComponent)
    if numericComponent == 0:
        return error
    else:
        return error / numericComponent


def testAllBiases():
    for biasNum in [2, 1, 0]:
        for biasIdx in range(len(params.biases[biasNum])):
            relError = relErrorInBias(biasNum, biasIdx)
            if relError > 10 ** (-4):
                print(f"entry {biasIdx} of bias {biasNum} failed test with rel error {relError}")
                return 
    

def testAllWeights():
    for weightNum in [2, 1, 0]:
        weight = params.weights[weightNum]
        for row in range(len(weight)):
            for col in range(len(weight[row])):
                relError = relErrorInWeight(weightNum, row, col)
                if relError > 10 ** (-4):
                    print(f"row, col {row}, {col} of weight {weightNum} failed test with rel error {relError}")
                    return

testAllBiases()
testAllWeights()