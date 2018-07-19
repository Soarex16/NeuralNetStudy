import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from nnet import NeuralNetwork

np.random.seed(1)


def remap(value, low1, high1, low2, high2):
    return low2 + (value - low1) * (high2 - low2) / (high1 - low1)


config = [2, 15, 10, 15, 1]
network = NeuralNetwork(config, learning_rate=0.0005)

Xn = 30
Yn = 30

X = np.linspace(-0.9, 0.9, Xn).reshape(-1, 1)
Y = np.linspace(-0.9, 0.9, Yn).reshape(-1, 1)


def f(x, y):
    return np.sin(3 * x) * np.cos(2 * y) * 0.5


X, Y = np.meshgrid(X, Y)
XY_train = np.stack((X.ravel(), Y.ravel()), axis=-1)

Z_train = f(X, Y).reshape(Xn * Yn, 1)
Z_train = remap(Z_train, np.min(Z_train), np.max(Z_train), 0.05, 0.95)

Z_ideal = f(X, Y)

epochs = 1000
err = []
for i in range(epochs):
    err.append(np.max(network.train(XY_train, Z_train)) / len(Z_train))
    if i % 100 == 0:
        print("running " + str(i) + " epoch")
print("error on end of train = " + str(err[-1]))

Z_predicted = network.predict(XY_train).reshape(Xn, Yn)
Z_predicted = remap(Z_predicted, 0.05, 0.95, np.min(Z_train), np.max(Z_train))

fig = plt.figure(0)
ax = fig.gca(projection='3d')
plot = ax.plot_surface(X, Y, Z_ideal, color='red', linewidth=1, rstride=5, cstride=5)
fig.suptitle('Input function', fontsize=16)

fig2 = plt.figure(1)
ax = fig2.gca(projection='3d')
plot2 = ax.plot_surface(X, Y, Z_predicted, color='gray', linewidth=0.5, rstride=5, cstride=5)
fig2.suptitle('Network approximation cfg:' + str(config) + "\n avg err = {:7f}".format(err[-1]), fontsize=16)

err_fig = plt.figure(3)
ax = err_fig.add_subplot(111)
ax.plot(range(epochs), err)

plt.show()
