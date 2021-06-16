from mnist import MNIST
import numpy as np
from FourLayerNet import FourLayerNet

mndata = MNIST("samples")
images, labels = mndata.load_training()
image = np.array(images[0])
print(image.shape)

net = FourLayerNet(len(images[0]), 16, 16, 10)
"""
A = np.array([
    [1, 2, 3], 
    [4, 5, 6]
])
B = np.array([
    [1 for m in range(3)]
    for l in range(4)
])
print(B.shape)
x = np.array([[10], [20], [30]])
# print(net.calcOutputLayer(images[0]))
"""