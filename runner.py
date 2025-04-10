from collections import Counter
from data import read_data
from nn import NN

import numpy as np
import sys


def fashion_transform(x):
    global fashion_labels
    return fashion_labels[x]


def emnist_transform(x):
    return chr(x + ord('A'))


def train(network: NN, x_train, y_train, lr=0.001, momentum=0.5, epochs=5):

    y_train = np.array(y_train).flatten()
    cost_per_epoch = Counter(range(1, epochs+1))
    for epoch in range(epochs):
        total_cost = 0
        indices = np.arange(len(x_train))
        np.random.shuffle(indices)
        x_train, y_train = x_train[indices], y_train[indices]
        for i in range(len(x_train)):
            total_cost += network.learn(x_train[i], y_train[i], lr, momentum)

        cost_per_epoch[epoch+1] = total_cost / len(x_train)
    print([f'{cost_per_epoch[i]:.4f}' for i in range(1, epochs+1)])


def test(network: NN, x_test, y_test, transform=(lambda x: x)):
    y_test = np.array(y_test).flatten()
    correct = 0
    accuracy = 0
    counter_size = np.max(y_train) + 1
    counts = Counter(range(counter_size))
    misses = Counter(range(counter_size))
    appearances = Counter(y_test)
    print([appearances[i] for i in range(counter_size)])

    flag = ""
    while not flag:
        flag = input('press key to continue... ')

    for i in range(len(x_test)):
        guess = network.classify(x_test[i])
        if guess == y_test[i]:
            counts[guess] += 1
            correct += 1
        else:
            misses[guess] += 1
        accuracy = correct / (i + 1)

    print(f"Total Correct: {correct}, accuracy of {accuracy:.4f}")
    print([(transform(i), counts[i]) for i in range(counter_size)])
    print([(transform(i), appearances[i]) for i in range(counter_size)])
    print('Which wrong answers were most common?')
    print([(transform(i), misses[i]) for i in range(counter_size)])
    print('Most missed answers:')
    print(sorted([(transform(i), appearances[i]-counts[i]) for i in range(counter_size)]))


if not sys.argv[1] in ['digits', 'fashion', 'emnist']:
    exit()


def transform(x): return x


fashion_labels = ['T-shirt/top',
                  'Trouser',
                  'Pullover',
                  'Dress',
                  'Coat',
                  'Sandal',
                  'Shirt',
                  'Sneaker',
                  'Bag',
                  'Ankle boot']

match sys.argv[1]:
    case 'fashion':
        transform = fashion_transform
    case 'emnist':
        transform = emnist_transform


dataset = sys.argv[1]
x_train, y_train, x_test, y_test = read_data(dataset, noisy_train=False, noisy_test=False)

match sys.argv[2]:
    case 'shallow':
        layer_sizes = [x_train[0].shape[1], 128, np.max(y_train)+1]

    case 'deep':
        layer_sizes = [x_train[0].shape[1], 256, 128, 64, 32, np.max(y_train)+1]


x_train = x_train / 255.0
x_test = x_test / 255.0
n = NN(layer_sizes)
train(n, x_train, y_train, lr=0.001, momentum=0.9, epochs=10)
test(n, x_test, y_test, transform=transform)
exit()
