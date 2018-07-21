import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from nnet import NeuralNetwork
import activation_functions as funcs

np.random.seed(1)


def remap(value, low1, high1, low2, high2):
    return low2 + (value - low1) * (high2 - low2) / (high1 - low1)


config = [2, 20, 15, 20, 1]
network = NeuralNetwork(config, learning_rate=0.00005, act_func=funcs.tanh, df_act_func=funcs.df_illogical)

Xn = 30
Yn = 30

X = np.linspace(-0.9, 0.9, Xn).reshape(-1, 1)
Y = np.linspace(-0.9, 0.9, Yn).reshape(-1, 1)


def f(x, y):
    return np.sin(3 * x) * np.cos(2 * y) * 0.5


X, Y = np.meshgrid(X, Y)
XY_train = np.stack((X.ravel(), Y.ravel()), axis=-1)

res = f(X, Y)

# shuffling dataset (I know, taht's awfull, I'll rewrite it later)
X_c = np.copy(X)
Y_c = np.copy(Y)
np.random.shuffle(X_c)
np.random.shuffle(Y_c)
XY_train = np.stack((X_c.ravel(), Y_c.ravel()), axis=-1)
res = f(X_c, Y_c)
#

Z_train = res.reshape(Xn * Yn, 1)
Z_train = remap(Z_train, np.min(Z_train), np.max(Z_train), 0.01, 0.99)

Z_ideal = f(X, Y)

epochs = 2000
err = []
for i in range(epochs):
    err.append(np.max(network.train(XY_train, Z_train)))
    if i % 100 == 0:
        print("running " + str(i) + " epoch")
print("error on end of train = " + str(err[-1]))

XY_pred = np.stack((X.ravel(), Y.ravel()), axis=-1)
Z_predicted = network.predict(XY_pred).reshape(Xn, Yn)  # XY_train
Z_predicted = remap(Z_predicted, 0.05, 0.95, np.min(Z_train), np.max(Z_train))

fig = plt.figure(0)
ax = fig.gca(projection='3d')
plot = ax.plot_wireframe(X, Y, Z_ideal, color='red', linewidth=1)
fig.suptitle('Input function', fontsize=16)

fig2 = plt.figure(1)
ax = fig2.gca(projection='3d')
plot2 = ax.plot_wireframe(X, Y, Z_predicted, color='gray', linewidth=0.5)
fig2.suptitle('Network approximation cfg:' + str(config) + "\n avg err = {:7f}".format(err[-1]), fontsize=16)

err_fig = plt.figure(2)
ax = err_fig.add_subplot(111)
ax.plot(range(epochs), err)

plt.show()
