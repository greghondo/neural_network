import numpy as np


def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    f = sigmoid(x)
    return f * (1-f)


def mse(true, pred):
    return ((true - pred)**2).mean()

class NeuralNetwork:
    def __init__(self, sizes):
        self.layers = len(sizes)
        self.sizes = sizes
        self.biases = (np.random.randn(x, 1) for x in len(sizes[1:]))
        self.weights = (np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:]))

    def feed_forward(self, x):
        for bias, weight in zip(self.biases, self.weights):
            x = sigmoid(np.dot(weight, x) + bias)
        return x


nn = NeuralNetwork()