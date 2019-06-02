import random
import numpy as np
import os
import datetime
import pickle
from activefunc import *
from costfunc import *


class Network(object):

    def __init__(self, sizes, cost=CrossEntropyCost):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.resDir = os.path.join(
            'trainResult', datetime.datetime.now().strftime("%Y-%m-%d-%H%M"))
        os.makedirs(self.resDir,exist_ok=True)
        self.cost = cost

    def large_weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def default_weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, modelName, lmbda=0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False
            ):
        if(evaluation_data):
            n_test = len(evaluation_data)
            hightestEpoch = (0, 0)
            evaluation_cost, evaluation_accuracy = [], []
        n = len(training_data)
        training_cost, training_accuracy = [], []
        print(
            "Epoch\ttraining_cost\ttraining_accuracy\tevaluation_cost\tevaluation_accuracy")
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            print("{:02}".format(j), end='|\t')
            if monitor_training_cost:
                self.monitor_cost(training_data, lmbda,
                                  training_cost, 'training', convert=False)
            if monitor_training_accuracy:
                self.monitor_accuracy(
                    training_data, training_accuracy, 'training', convert=True)
            if evaluation_data:
                if monitor_evaluation_cost:
                    self.monitor_cost(evaluation_data, lmbda,
                                      evaluation_cost, 'evaluation', convert=True)
                if monitor_evaluation_accuracy:
                    self.monitor_accuracy(
                        evaluation_data, evaluation_accuracy, 'evaluation', convert=False)
            self.save_train_result(j, modelName)
            print('')
        return training_cost, training_accuracy, evaluation_cost, evaluation_accuracy

    def monitor_cost(self, data, lmbda, costs, dataType, convert):
        cost = self.total_cost(data, lmbda, convert)
        costs.append(cost)
        print('{:.4f}'.format(cost), end='|\t')

    def monitor_accuracy(self, data, accuracys, dataType, convert):
        accuracy = self.accuracy(data, convert)
        accuracys.append(accuracy)
        print('{:.2%}'.format(accuracy), end='|\t')

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
        delta = self.cost.delta(zs[-1], activations[-1], y)
        nabla_b[-1] = np.sum(delta, axis=1, keepdims=True)
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta)*sp
            nabla_b[-l] = np.sum(delta, axis=1, keepdims=True)
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        tag = [y for x, y in test_data]
        data = np.concatenate([x for x, y in test_data],  axis=1),
        test_results = np.argmax(self.feedforward(data), axis=0)
        return sum(int(x == y) for x, y in zip(test_results, tag))

    def accuracy(self, input_data, convert=False):
        tag = [np.argmax(y) if convert else y for x, y in input_data]
        data = np.concatenate([x for x, y in input_data],  axis=1)
        results = np.argmax(self.feedforward(data), axis=0)
        return sum(int(x == y) for x, y in zip(results, tag))/len(input_data)

    def total_cost(self, input_data, lmbda, convert=False):
        cost = 0.0
        n_data = len(input_data)
        y = np.concatenate([self.vectorized_result(
            y) if convert else y for x, y in input_data],  axis=1)
        data = np.concatenate([x for x, y in input_data],  axis=1)
        a = self.feedforward(data)

        cost += self.cost.fn(a, y)/n_data
        cost += 0.5*(lmbda/n_data) * \
            sum(np.linalg.norm(w)**2 for w in self.weights)
        return cost

    def vectorized_result(self, y):
        e = np.zeros((self.sizes[-1], 1))
        e[y] = 1.0
        return e
