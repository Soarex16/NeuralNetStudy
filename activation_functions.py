# activation functions and their derivatives
import numpy as np


def identity(x):
    return x


def df_identity(x):
    return 1


def bin_step(x):
    return np.where(x < 0, 0, 1)


def df_bin_step(x):
    return np.where(x == 0, None, 0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def df_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def tanh(x):
    return np.tanh(x)


def df_tanh(x):
    return 1 - np.power(tanh(x), 2)


def arctan(x):
    return np.arctan(x)


def df_arctan(x):
    return 1 / (1 + np.power(x, 2))


def relu(x):
    return np.maximum(0, x)


def df_relu(x):
    x_ar = np.array(x)
    return np.where(x_ar < 0, 0, 1)


def softplus(x):
    return np.log(1 + np.exp(x))


def df_softplus(x):
    return 1 / (1 + np.exp(-x))


# I don't know why it better
def df_illogical(x):
    return 1 - np.power(x, 2)

# def softmax(x):
#     return np.exp(x) / (np.sum(np.exp(x)))
#
#
# def stable_softmax(x):
#     shiftx = x - np.max(x)
#     exps = np.exp(shiftx)
#     return exps / np.sum(exps)
#
# test = [1.2, -1, 1.6, 5, 7]
# a = np.asarray(test)
# print(df_relu(a))