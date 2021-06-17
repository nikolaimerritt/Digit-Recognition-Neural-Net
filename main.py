import numpy as np
from mnist import MNIST
from Params import Params, paramsSum

mndata = MNIST("samples")
images, lables = mndata.load_training()

A = np.array([
    [1, 2, 3],
    [4, 5, 6]
])
x = np.array([1, 2, 3])

params = Params([A], [x])

paramsList = [Params([n * A], [n * x]) for n in [1, 2, 3]]
print(sum(paramsList) / len(paramsList))

#net = FourLayerNet(len(images[0]), 16, 16, 10)