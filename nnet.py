import numpy as np
import activation_functions as funcs


# Maps data from [low1, high1] => [low2, high2]
def remap(value, low1, high1, low2, high2):
    return low2 + (value - low1) * (high2 - low2) / (high1 - low1)


# NOTE: INPUT MUST BE COLUMN VECTOR
class NeuralNetwork:

    def __init__(self, config, learning_rate=0.1, act_func=funcs.sigmoid, df_act_func=funcs.df_sigmoid):
        self.config = config
        self.synapses = len(config) - 1
        self.learning_rate = learning_rate

        self.accuracy = 0.0
        self.mse = -1

        if callable(act_func) and callable(df_act_func):
            self.act_func = act_func
            self.df_act_func = df_act_func
        else:
            raise ValueError("Activation function (and their derivative) must be callable \nActivation "
                             "function type is " + str(type(act_func)) + "\nActivation function derivative type is " +
                             str(type(df_act_func)))

        self.weights = []
        self.bias = []
        self._init_weights()

    # TODO: add keras-like layers adding (with their own activation functions) --Do I need this? Maybe, no
    # TODO: what about differencing layers (again, keras-line) UPD: but I don't know about other layers types

    # Deprecated functionality without ability to change activation function
    # def activation_function(self, x):
    #     return 1 / (1 + np.exp(-x))
    #
    # def activation_function_derr(self, x):
    #     return self.activation_function(x) * (1 - self.activation_function(x))
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
            signal = self.act_func(z)
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

            # here must be derivative on sigmoid, but 1 - sigmoid(x^2) better
            # that's strange, but it works...okay, fine
            # is this hack always better?
            # tested on MNIST and function approximation and it's better :/
            d_act_func = self.df_act_func(outputs[i + 1])
            # d_act_func = 1 - np.power(outputs[i + 1], 2)
            # d_act_func = outputs[i + 1] * (1 - outputs[i + 1])
            self.weights[i] += self.learning_rate * np.dot(err[-1] * d_act_func, outputs[i].T)
            self.bias[i] += self.learning_rate * err[-1]

        return err[0]

    def predict(self, x):
        return self._forward(x)[0]

    def train(self, x, y):
        return self._backprop(x, y)

    # Evaluates one epoch of learning and returns max error on epoch
    def run_epoch(self, x_train, y_train):
        epoch_err = []
        for i in range(np.size(x_train)):
            epoch_err.append(self.train(x_train[i], y_train[i]))

        epoch_err = np.array(epoch_err)
        self.mse = np.sum(np.square(epoch_err))

        return epoch_err.max()

    # counts accuracy and mse
    def validate(self, X_test, Y_test):
        accuracy = 0
        for i in range(len(X_test)):
            accuracy += 1 if np.equal(self.predict(X_test[i]), Y_test[i]).all() else 0

        self.accuracy = accuracy / len(Y_test)

        return self.accuracy

    # It is assumed that the data are submitted pre-processed to numpy array with correct shape
    def fit(self, X_train, Y_train, epochs, verbose=True, verbose_period=100):
        error = []
        for epoch in range(epochs):
            if verbose and not (epoch % verbose_period):
                print("Running %f epoch".format(epoch))

            error.append(self.run_epoch(X_train, Y_train))

        return np.array(error)

    # just same as fit, but prepares data before training
    def fit_raw(self, X_raw, Y_raw, epochs, verbose, verbose_period):
        x_min = np.min(X_raw)
        x_max = np.max(X_raw)
        y_min = np.min(Y_raw)
        y_max = np.max(Y_raw)
        x_mapped = remap(X_raw, x_min, x_max, 0.01, 0.99)
        y_mapped = remap(Y_raw, y_min, y_max, 0.01, 0.99)

        self.fit(x_mapped, y_mapped, epochs, verbose, verbose_period)
