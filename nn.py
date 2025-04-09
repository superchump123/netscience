import sys
from collections import Counter
from data import read_data
import numpy as np


class Layer:

    @staticmethod
    def relu(val):
        return np.maximum(0, val)

    @staticmethod
    def d_relu(val):
        return (val > 0).astype(float)

    @staticmethod
    def sigmoid(val):
        return 1 / (1 + np.exp(-val))

    @staticmethod
    def softmax(val):
        val = val - np.max(val)
        e = np.exp(val)
        return e / e.sum()

    @staticmethod
    def d_sigmoid(val):
        s = Layer.sigmoid(val)
        return s * (1 - s)

    @staticmethod
    def d_softmax(val):
        return np.ones_like(val)

    def __init__(self, nodes_in, nodes_out, activation=relu):
        self.nodes_in = nodes_in
        self.nodes_out = nodes_out
        self.activation = activation
        self.d_activation = None
        self.weights = None

        match self.activation:
            case Layer.relu:
                self.d_activation = Layer.d_relu
                stddev = np.sqrt(2 / nodes_in)
                self.weights = np.random.randn(nodes_in, nodes_out) * stddev
            case Layer.sigmoid:
                limit = np.sqrt(6 / (nodes_in + nodes_out))
                self.weights = np.random.uniform(-limit, limit, (nodes_in, nodes_out))
                self.d_activation = Layer.d_sigmoid
            case Layer.softmax:
                limit = np.sqrt(6 / (nodes_in + nodes_out))
                self.weights = np.random.uniform(-limit, limit, (nodes_in, nodes_out))
                self.d_activation = Layer.d_softmax

        self.biases = np.zeros(nodes_out)

        self.w_velocities = np.zeros_like(self.weights)
        self.b_velocities = np.zeros_like(self.biases)

        self.last_z = np.zeros(nodes_in)
        self.last_input = np.zeros(nodes_in)

    def forward(self, inputs):
        inputs = np.array(inputs).flatten()
        self.last_input = inputs
        z = np.dot(inputs, self.weights) + self.biases
        self.last_z = z
        return self.activation(z)

    def apply_gradient(self, lr, momentum, grad_w, grad_b):
        self.w_velocities = momentum * self.w_velocities - lr * grad_w
        self.b_velocities = momentum * self.b_velocities - lr * grad_b

        self.weights += self.w_velocities
        self.biases += self.b_velocities


class NN:
    def __init__(self, layer_sizes):
        self.layers: list[Layer] = []
        for i in range(len(layer_sizes) - 1):
            if i == len(layer_sizes) - 2:
                self.layers.append(Layer(layer_sizes[i], layer_sizes[i+1], Layer.softmax))
            else:
                self.layers.append(Layer(layer_sizes[i], layer_sizes[i+1], Layer.relu))

    def calculate_outputs(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def classify(self, input):
        outputs = self.calculate_outputs(input)
        return np.argmax(outputs)

    def _apply_gradients(self, learn, momentum, grad_w, grad_b):
        for i, layer in enumerate(self.layers):
            layer.apply_gradient(learn, momentum, grad_w[i], grad_b[i])

    def _cost(self, x, expected_label):
        output = self.calculate_outputs(x)
        epsilon = 1e-12  # to avoid log(0)
        return -np.log(output[expected_label] + epsilon)

    def _avg_cost(self, x_batch, y_batch):
        total = sum(self._cost(x, y) for x, y in zip(x_batch, y_batch))
        return total / len(x_batch)

    def backprop(self, x, y):
        output = self.calculate_outputs(x)
        expected = np.zeros_like(output)
        expected[y] = 1
        last = self.layers[-1]
        delta = output - expected

        grad_w = [None] * len(self.layers)
        grad_b = [None] * len(self.layers)

        grad_w[-1] = np.outer(last.last_input, delta)
        grad_b[-1] = delta

        for i in range(2, len(self.layers) + 1):
            layer = self.layers[-i]
            next_layer = self.layers[-i + 1]

            z = layer.last_z
            dz = layer.d_activation(z)

            delta = np.dot(next_layer.weights, delta) * dz

            grad_w[-i] = np.outer(layer.last_input, delta)
            grad_b[-i] = delta

        return grad_w, grad_b

    def learn(self, x, y, lr, momentum):
        grad_w, grad_b = self.backprop(x, y)
        self._apply_gradients(lr, momentum, grad_w, grad_b)
