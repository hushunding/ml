import random
import numpy as np
import os
import datetime
import pickle


class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        self.resDir = os.path.join(
            'trainResult', datetime.datetime.now().strftime("%Y-%m-%d-%H%M"))
        os.makedirs(self.resDir)

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, modelName, test_data=None):
        if(test_data):
            n_test = len(test_data[1])
            hightestEpoch = (0, 0)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                evaluateRes = self.evaluate(test_data)
                if evaluateRes > hightestEpoch[1]:
                    hightestEpoch = (j, evaluateRes)
                print("Epoch {0}: {1} / {2}".format(j, evaluateRes, n_test))

            else:
                print("Epoch {0} complete".format(j))
            self.save_train_result(j, modelName)
        if(test_data):
            j, evaluateRes = hightestEpoch
            print("best epoch: {0}: {1} / {2}".format(j, evaluateRes, n_test))

    def save_train_result(self, j, modelName):
        datapath = os.path.join(self.resDir, "{2}_Epoch_{0}_{1}.data".format(
            "_".join([str(i) for i in self.sizes]), j, modelName))
        pickle.dump((self.biases, self.weights), open(datapath, 'wb'))

    def update_mini_batch(self, mini_batch, eta):
        n = len(mini_batch)
        X = np.concatenate([x for x, y in mini_batch], axis=1)
        Y = np.concatenate([y for x, y in mini_batch], axis=1)
        # 反向传播，误差
        nabla_b, nabla_w = self.backprop(X, Y)

        self.biases = [b - eta/n*nb for b, nb in zip(self.biases, nabla_b)]
        self.weights = [w - eta/n*nw for w, nw in zip(self.weights, nabla_w)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # 前向传播，并记录各层输出
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # 反向传播
        delta = self.cost_derivative(activations[-1], y)*sigmoid_prime(zs[-1])
        nabla_b[-1] = np.sum(delta, axis=1, keepdims=True)
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta)*sp
            nabla_b[-l] = np.sum(delta, axis=1, keepdims=True)
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    # 代价函数 导数
    def cost_derivative(self, output_activations, y):
        return output_activations - y

    def evaluate(self, test_data):
        data,tag = test_data
        test_results = np.argmax(self.feedforward(data),axis=0)
        return sum(int(x == y) for x, y in zip(test_results,tag))


def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))
