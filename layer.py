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
