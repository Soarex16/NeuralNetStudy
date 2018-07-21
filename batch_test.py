import numpy as np
import matplotlib.pyplot as plt
from nnet import NeuralNetwork
import activation_functions as funcs

config = [1, 5, 5, 1]
net = NeuralNetwork(config, learning_rate=0.1, act_func=funcs.sigmoid, df_act_func=funcs.df_sigmoid)


def f(x):
    return x ** 2


vf = np.vectorize(f)

x = np.array(np.linspace(-1, 1, num=10), ndmin=2)
y = vf(x)

epochs = 5000
err = net.fit_raw(x, y, epochs, True, 10)

test_sample = np.array([0.1])
y_pred = net.predict(test_sample)

plt.plot(np.array(range(epochs) + 1), np.array(err))
plt.show()
