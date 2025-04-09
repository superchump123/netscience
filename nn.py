from collections import Counter
from data import read_data
import numpy as np

input_path = 'data/'
x_train, y_train, x_test, y_test = read_data(input_path)
layer_sizes = [x_train[0].shape[1], 128, 64, 10]


class Layer:
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

    def __init__(self, nodes_in, nodes_out, activation=sigmoid):
        self.nodes_in = nodes_in
        self.nodes_out = nodes_out

        # Xavier initialization for weights
        limit = np.sqrt(6 / (nodes_in + nodes_out))
        self.weights = np.random.uniform(-limit, limit, (nodes_in, nodes_out))
        self.biases = np.zeros(nodes_out)

        self.last_z = np.zeros(nodes_in)

        self.activation = activation
        if activation == Layer.sigmoid:
            self.d_activation = Layer.d_sigmoid
        else:
            self.d_activation = Layer.d_softmax

        self.last_input = np.zeros(nodes_in)

    def forward(self, inputs):
        inputs = np.array(inputs).flatten()
        self.last_input = inputs
        z = np.dot(inputs, self.weights) + self.biases
        self.last_z = z
        return self.activation(z)

    def apply_gradient(self, lr, grad_w, grad_b):
        self.weights -= lr * grad_w
        self.biases -= lr * grad_b


class NN:
    def __init__(self, layer_sizes):
        self.layers: list[Layer] = []
        for i in range(len(layer_sizes) - 1):
            if i == len(layer_sizes) - 2:
                self.layers.append(Layer(layer_sizes[i], layer_sizes[i+1], Layer.softmax))
            else:
                self.layers.append(Layer(layer_sizes[i], layer_sizes[i+1]))

    def calculate_outputs(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def classify(self, input):
        outputs = self.calculate_outputs(input)
        return np.argmax(outputs)

    def _apply_gradients(self, learn, grad_w, grad_b):
        for i, layer in enumerate(self.layers):
            layer.apply_gradient(learn, grad_w[i], grad_b[i])

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

    def learn(self, x, y, lr):
        grad_w, grad_b = self.backprop(x, y)
        self._apply_gradients(lr, grad_w, grad_b)


def train(network: NN, x_train, y_train, x_test, y_test, lr=0.01, epochs=5, report_every=1000):

    y_train = np.array(y_train).flatten()
    y_test = np.array(y_test).flatten()
    cost_per_epoch = Counter(range(1, epochs+1))
    for epoch in range(epochs):
        total_cost = 0
        indices = np.arange(len(x_train))
        np.random.shuffle(indices)
        x_train, y_train = x_train[indices], y_train[indices]
        for i in range(len(x_train)):
            network.learn(x_train[i], y_train[i], lr)
            total_cost += network._cost(x_train[i], y_train[i])

            if (i + 1) % report_every == 0 or i == len(x_train) - 1:
                avg_so_far = total_cost / (i + 1)
                print(f"  Progress: {i+1}/{len(x_train)} — Avg Cost: {avg_so_far:.4f}")

        print(f"Epoch {epoch+1}, Avg Cost: {total_cost / len(x_train)}")
        cost_per_epoch[epoch+1] = total_cost / len(x_train)

    print([cost_per_epoch[i] for i in range(1, epochs+1)])
    correct = 0
    accuracy = 0
    counts = Counter(range(10))
    appearances = Counter(y_test)
    print([appearances[i] for i in range(10)])

    flag = ""
    while not flag:
        flag = input('press key to continue... ')

    for i in range(len(x_test)):
        guess = network.classify(x_test[i])
        if guess == y_test[i]:
            counts[guess] += 1
            correct += 1
        else:
            print(f'Missed picture {i}, output {guess} when answer was {y_test[i]}')
        accuracy = correct / (i + 1)
        if (i + 1) % report_every == 0 or i == len(x_test) - 1:
            print(f"Current accuracy: {accuracy:.4f}")
    print(f"Total Correct: {correct}, accuracy of {accuracy:.4f}")
    print([counts[i] for i in range(10)])
    print([appearances[i] for i in range(10)])


x_train = x_train / 255.0
x_test = x_test / 255.0
n = NN(layer_sizes)
train(n, x_train, y_train, x_test, y_test, lr=0.001,epochs=10)
