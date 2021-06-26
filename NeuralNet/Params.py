from typing import List
import numpy as np
from NeuralNet import util


class Params:
    ZERO = 0
    
    def __init__(self, weights: List[np.ndarray], biases: List[np.ndarray]):
        self.weights = weights
        self.biases = biases
        self.count = len(self.weights)
    

    def saveToFile(self, folder="params-values"):
        for i in range(len(self.weights)):
            np.savetxt(f"{folder}/weights/weight{i}.txt", self.weights[i])
        
        for i in range(len(self.biases)):
            np.savetxt(f"{folder}/biases/bias{i}.txt", self.biases[i])


    def __add__(self, other):
        if other == Params.ZERO:
            return self
            
        newWeights = [myWeight + theirWeight for myWeight, theirWeight in zip(self.weights, other.weights)]
        newBiases = [myBias + theirBias for myBias, theirBias in zip(self.biases, other.biases)]
        return Params(newWeights, newBiases)
    

    def __sub__(self, other):
        return self + (-1) * other
    

    def __radd__(self, other):
        if other == Params.ZERO:
            return self
        else:
            return other + self
    

    def __rmul__(self, scalar):
        newWeights = [scalar * weight for weight in self.weights]
        newBiases = [scalar * bias for bias in self.biases]
        return Params(newWeights, newBiases)
    

    def __truediv__(self, scalar):
        newWeights = [weight / scalar for weight in self.weights]
        newBiases = [bias / scalar for bias in self.biases]
        return Params(newWeights, newBiases)
    

    def squaredNorm(self):
        return sum((weight * weight).sum() for weight in self.weights) + sum((bias * bias).sum() for bias in self.biases)
    

    @classmethod
    def random(cls, sizes: List[int]):
        weightDims = [(sizes[i+1], sizes[i]) for i in range(len(sizes) - 1)]
        biasDims = sizes[1 : len(sizes)]

        weights = [util.unbiasedMatrix(*dims) for dims in weightDims]
        biases = [util.unbiasedVector(dim) for dim in biasDims]
        return cls(weights, biases)


    @classmethod
    def loadFromFile(cls, folder="params-values"):
        weights = []
        for weightFile in sorted(util.txtsInFolder(f"{folder}/weights")):
            weights.append(np.genfromtxt(f"{folder}/weights/{weightFile}"))
        
        biases = []
        for biasFile in sorted(util.txtsInFolder(f"{folder}/biases")):
            biases.append(np.genfromtxt(f"{folder}/biases/{biasFile}"))
        
        return cls(weights, biases)