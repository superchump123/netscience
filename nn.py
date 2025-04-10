import numpy as np
from layer import Layer


class NN:
    def __init__(self, layer_sizes):
        self.layers: list[Layer] = []
        for i in range(len(layer_sizes) - 1):
            if i == len(layer_sizes) - 2:
                self.layers.append(Layer(layer_sizes[i], layer_sizes[i+1], Layer.softmax))
            else:
                self.layers.append(Layer(layer_sizes[i], layer_sizes[i+1], Layer.relu))

        self.grad_w = [np.zeros_like(layer.weights) for layer in self.layers]
        self.grad_b = [np.zeros_like(layer.biases) for layer in self.layers]

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

    def cost(self, x, expected_label):
        output = self.calculate_outputs(x)
        epsilon = 1e-12  # to avoid log(0)
        return -np.log(output[expected_label] + epsilon)

    def backprop(self, x, y):
        output = self.calculate_outputs(x)
        expected = np.zeros_like(output)
        expected[y] = 1

        cost = -np.log(output[y] + 1e-12)

        last = self.layers[-1]
        delta = output - expected

        self.grad_w[-1][:] = np.outer(last.last_input, delta)
        self.grad_b[-1][:] = delta

        for i in range(2, len(self.layers) + 1):
            layer = self.layers[-i]
            next_layer = self.layers[-i + 1]

            z = layer.last_z
            dz = layer.d_activation(z)

            delta = np.dot(next_layer.weights, delta) * dz

            self.grad_w[-i][:] = np.outer(layer.last_input, delta)
            self.grad_b[-i][:] = delta

        return self.grad_w, self.grad_b, cost

    def learn(self, x, y, lr, momentum):
        grad_w, grad_b, cost = self.backprop(x, y)
        self._apply_gradients(lr, momentum, grad_w, grad_b)
        return cost
