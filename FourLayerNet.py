from typing import List
import numpy as np

def relu(data: List[float]) -> List[float]:
    return [0 if x < 0 else x for x in data]


def sigmoid(data: List[float]) -> List[float]:
    return [1 / (1 + np.exp(-x)) for x in data]

def normalise(data: List[float]) -> List[float]:
    norm = sum(x ** 2 for x in data) ** (1/2)
    return [x / norm for x in data]


class FourLayerNet:
    def __init__(self, inputLayerSize: int, layer1Size: int, layer2Size: int, outputLayerSize: int) -> None:
        self.fstWeights = np.random.rand(layer1Size, inputLayerSize)
        self.sndWeights = np.random.rand(layer2Size, layer1Size)
        self.thdWeights = np.random.rand(outputLayerSize, layer2Size)

    
    def calcOutputLayer(self, inputLayer):
        layer1 = relu(self.fstWeights @ inputLayer)
        layer2 = relu(self.sndWeights @ layer1)
        output = self.thdWeights @ layer2
        return normalise(output)
