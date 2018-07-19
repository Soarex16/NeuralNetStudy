import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:

    def __init__(self, config, learning_rate=0.1):
        self.config = config
        self.synapses = len(config) - 1
        self.learning_rate = learning_rate

        self.weights = []
        self.bias = []
        self._init_weights()

    # TODO: add ability to change activation function (with derivatives)
    # TODO: add keras-like layers adding (with their own activation functions) --Do I need this? Maybe, no
    # TODO: what about differencing layers (again, keras-line) UPD: but I don't know about other layers types
    def activation_function(self, x):
        return 1 / (1 + np.exp(-x))

    def activation_function_derr(self, x):
        return self.activation_function(x) * (1 - self.activation_function(x))

    def _init_weights(self):
        for i in range(self.synapses):
            self.weights.append(np.random.rand(self.config[i + 1], self.config[i]) - 0.5)
            # self.weights.append(np.random.normal(0.0, pow(self.config[i], -0.5), (self.config[i + 1], self.config[
            # i])) - 0.5)
            self.bias.append(np.random.rand() - 0.5)

    def _forward(self, inputs):
        signal = np.array(inputs, ndmin=2).T
        outputs = [signal]

        for i in range(self.synapses):
            z = np.dot(self.weights[i], signal) + self.bias[i]
            signal = self.activation_function(z)
            outputs.append(signal)

        return signal, outputs

    def _backprop(self, x, y_mean):
        y_pred, outputs = self._forward(x)

        y_target = np.array(y_mean, ndmin=2).T

        err_out = (y_target - y_pred)

        err = [err_out]

        self.weights[-1] += self.learning_rate * np.dot(err_out * outputs[-1] * (1 - outputs[-1]),
                                                        np.transpose(outputs[-2]))
        self.bias[-1] += self.learning_rate * err[-1]

        for i in reversed(range(self.synapses - 1)):
            hidden_err = np.dot(self.weights[i + 1].T, err[-1])
            err.append(hidden_err)

            # here must be derivative on sigmoid, but 1 - x^2 better
            # that's strange, but it works...okay, fine
            # is this hack always better?
            d_act_func = 1 - np.power(outputs[i + 1], 2)  # outputs[i + 1] * (1 - outputs[i + 1])
            self.weights[i] += self.learning_rate * np.dot(err[-1] * d_act_func, outputs[i].T)
            self.bias[i] += self.learning_rate * err[-1]

        return err[0]

    def predict(self, x):
        return self._forward(x)[0]

    def train(self, x, y):
        return self._backprop(x, y)
