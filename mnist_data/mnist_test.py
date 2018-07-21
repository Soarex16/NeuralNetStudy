import numpy as np
import matplotlib.pyplot as plt
from nnet import NeuralNetwork
import activation_functions as funcs

np.random.seed(0)


def remap(value, low1, high1, low2, high2):
    return low2 + (value - low1) * (high2 - low2) / (high1 - low1)


data_file = open("../mnist_data/mnist_train_100.csv")
data_list = data_file.readlines()
data_file.close()

config = [28 * 28, 200, 10]
net = NeuralNetwork(config, learning_rate=0.05, act_func=funcs.sigmoid, df_act_func=funcs.df_sigmoid)

X = []
Y = []
for data in data_list:
    raw_values = data.split(',')
    label = raw_values[0]
    x = np.asfarray(raw_values[1:])
    x = remap(x, 0, 255, 0.01, 0.99)

    y = np.zeros(10) + 0.01
    y[int(label)] = 0.99

    X.append(x)
    Y.append(y)

X = np.array(X)
Y = np.array(Y)

epochs = 80 # after 30 ne nado
err = []
for i in range(epochs):
    epoch_err = []
    for train_idx in range(len(Y)):
        epoch_err.append(net.train(X[train_idx], Y[train_idx]))
    err.append(np.max(np.array(epoch_err)))
    # err.append(np.max(net.train(X, Y)) / len(Y))
    if i % 100 == 0:
        print("running " + str(i) + " epoch")

print("error on end of train = " + str(err[-1]))

validate_file = open("../mnist_data/mnist_test_10.csv")
validate_list = validate_file.readlines()
validate_file.close()

scorecard = []
for i in range(len(validate_list)):
    raw_values = validate_list[i].split(',')
    label = raw_values[0]
    x = np.asfarray(raw_values[1:])
    x = remap(x, 0, 255, 0.01, 0.99)

    y = np.zeros(10) + 0.01
    y[int(label)] = 0.99

    y_pred = net.predict(x)

    # plt.imshow(x.reshape((28, 28)), cmap='Greys', interpolation=None)
    # plt.suptitle(label)
    # plt.show()
    # print("res " + str(y_pred))
    pred_num = np.argmax(y_pred)

    if pred_num == int(label):
        scorecard.append(1)
    else:
        scorecard.append(0)

    print(label + " predicted as " + str(pred_num))

# accuracy
np_scard = np.asarray(scorecard)
accuracy = np_scard.sum() / np_scard.size
print('accuracy = ' + str(accuracy))

err_fig = plt.figure(1)
ax = err_fig.add_subplot(111)
ax.plot(range(epochs), err)
plt.show()

# shows what inside network :)
for weight in net.weights:
    plt.imshow(weight, cmap='Greys', interpolation=None)
    plt.show()
# fig = plt.figure(0)
# ax = fig.add_subplot(111)
#
# idx = 5
# # make input
# x = X[idx]
# label = data_list[idx].split(',')[0]
#
# plt.imshow(x.reshape((28, 28)), cmap='Greys', interpolation=None)
# fig.suptitle(label)
#
# # get i-th column (because we feed all dataset at epoch (i can feed dataset per sample))
# y_pred = net.predict(x)[:,idx]
# print("raw net out " + str(y_pred))
# print("predicted " + str(np.argmax(y_pred)))
#
# # generate example result
# y = np.zeros(10) + 0.01
# y[int(label)] = 0.99
# print(y)
#
# err_fig = plt.figure(1)
# ax = err_fig.add_subplot(111)
# ax.plot(range(epochs), err)
#
# plt.show()

# print("-------")
# print(X[0])
# print(Y[0])
