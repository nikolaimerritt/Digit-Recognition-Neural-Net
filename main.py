from mnist import MNIST
import numpy as np
from FourLayerNet import FourLayerNet

mndata = MNIST("samples")
images, labels = mndata.load_training()
image = np.array(images[0])


net = FourLayerNet(len(images[0]), 16, 16, 10)
desiredOutputLayer = np.array(list(range(10)))
net.testCostGradient(images[0], desiredOutputLayer)