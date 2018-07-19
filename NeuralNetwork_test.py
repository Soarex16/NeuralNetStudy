import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from NeuralNetwork import NeuralNetwork

config = [2, 20, 20, 20, 1]
network = NeuralNetwork(config, learning_rate=0.01)


Xn = 100
Yn = 100

X = np.linspace(-0.5, 0.5, Xn).reshape(-1, 1)
Y = np.linspace(-0.5, 0.5, Yn).reshape(-1, 1)


def f(x, y):
    return np.abs(0.5 * np.sin(3*x) * np.cos(2*y))

X, Y = np.meshgrid(X, Y)
XY_train = np.stack((X.ravel(), Y.ravel()), axis=-1)
Z_train = f(X, Y).reshape(Xn * Yn, 1)

Z_ideal = f(X, Y)

epochs = 1000
err = []
for i in range(epochs):
    print("running " + str(i) + " epoch")
    err.append( network.train(XY_train, Z_train) )
Z_predicted = network.predict(XY_train).reshape(Xn, Yn)

fig = plt.figure(0)
ax = fig.gca(projection='3d')
plot = ax.plot_surface(X, Y, Z_ideal, color='red', linewidth=1, rstride=5, cstride=5)
fig.suptitle('Input function', fontsize=16)

fig2 = plt.figure(1)
ax = fig2.gca(projection='3d')
plot2 = ax.plot_surface(X, Y, Z_predicted, color='gray', linewidth=0.5, rstride=5, cstride=5)
fig2.suptitle('Network approximation', fontsize=16)

plt.show()
#
# # чему хотим обучить сеть
# def target_func(x, y):
#     return x * y
#
#
# # размер обучающей выборки
# samples_num = 10
#
# X = np.linspace(0.01, 0.99, samples_num).reshape(-1, 1)
# Y = np.linspace(0.01, 0.99, samples_num).reshape(-1, 1)
# Xm, Ym = np.meshgrid(X, Y)
# XY = np.stack( (Xm.ravel(), Ym.ravel()), axis=-1 )
# Z = target_func(Xm, Ym).reshape(samples_num * samples_num, 1)
#
# # тут будем хранить ошибку сети
# net_err = []
#
# num_epochs = 1
#
# # учими считаем среднюю ошибку на весь датасет за эпоху
# for i in range(num_epochs):
#     print("running epoch #" + str(i))
#     ep_err = 0
#     for j in range(len(XY)):
#         ep_err += np.asscalar( net.train(XY[j], Z[j]) )
#     net_err.append(np.abs(ep_err / samples_num))
#
# # print(net.weights)
#
# # plt.plot(range(samples_num * num_epochs), net_err)
# y_pred = [[]]
# for i in range(samples_num):
#     for j in range(samples_num):
#         y_pred[j][i] = np.asscalar( net.predict([X[j], Y[i]]) )
#     # print("x = " + str(x) + " y = " + str(y_pred[-1]))
#
# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# axes = fig.add_subplot(211, projection='3d')
# err_plt = fig.add_subplot(212)
#
#
# axes.plot_wireframe(X, Y, np.array(y_pred))
# axes.plot_wireframe(X, Y, Z)
#
# err_plt.plot(range(1, num_epochs + 1), net_err)
# plt.show()