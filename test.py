from mxnet.gluon import nn
from mxnet import autograd, init, np, npx
npx.set_np()

conv2d = nn.Conv2D(channels=1, kernel_size=(1,2))
conv2d.initialize()

X = np.ones((6, 8))
X[:, 2:6] = 0
X = X.reshape(1, 1, 6, 8)