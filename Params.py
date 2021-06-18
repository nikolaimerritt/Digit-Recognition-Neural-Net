from typing import List
import numpy as np

class Params:
    ZERO = 0
    def __init__(self, weights: List[np.ndarray], biases: List[np.ndarray]):
        self.weights = weights
        self.biases = biases
    
    def saveToFile(self, folder="params-values"):
        np.savez(f"{folder}/weights.npz", *self.weights)
        np.savez(f"{folder}/biases.npz", *self.biases)
    
    
    def loadFromFile(self, folder="params-values"):
        self.weights = list(np.load(f"{folder}/weights.npz").values())
        self.biases = list(np.load(f"{folder}/biases.npz").values())
    
    def __add__(self, other):
        newWeights = [myWeight + theirWeight for myWeight, theirWeight in zip(self.weights, other.weights)]
        newBiases = [myBias + theirBias for myBias, theirBias in zip(self.biases, other.biases)]
        return Params(newWeights, newBiases)
    
    def __radd__(self, other):
        if other == 0 or other == Params.ZERO:
            return self
        else:
            return other + self
    
    def __mul__(self, scalar):
        newWeights = [scalar * weight for weight in self.weights]
        newBiases = [scalar * bias for bias in self.biases]
        return Params(newWeights, newBiases)
    
    def __truediv__(self, scalar):
        newWeights = [weight / scalar for weight in self.weights]
        newBiases = [bias / scalar for bias in self.biases]
        return Params(newWeights, newBiases)
    
    def squaredNorm(self):
        return sum((weight * weight).sum() for weight in self.weights) + sum((bias * bias).sum() for bias in self.biases)


def paramsSum(paramsList: List[Params]) -> Params:
    start = paramsList[0]
    return sum(paramsList[1:], start=start)