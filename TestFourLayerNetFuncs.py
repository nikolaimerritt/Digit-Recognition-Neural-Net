from numpy.core.numeric import outer
import FourLayerNet
import numpy as np
import Params

x = np.array([1, -1, 2])
A = np.array([
    [1, 2, -3],
    [4, -1.5, 2]
])
params = FourLayerNet.getRandomParams(3, 4, 5, 6)

outLayer = np.array([8, 9, -7.1, 8])
desOutLayer = np.array([9, 1, 5.1, -2.8])

print(FourLayerNet.squaredNorm(outLayer - desOutLayer))