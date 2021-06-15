from mnist import MNIST
from FourLayerNet import FourLayerNet

mndata = MNIST("samples")
images, labels = mndata.load_training()

net = FourLayerNet(len(images[0]), 16, 16, 10)
print(net.calcOutputLayer(images[0]))