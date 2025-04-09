from collections import Counter
from data import read_data
from nn import NN

import numpy as np
import sys


def l_report(x):
    return chr(x + ord('A'))


def train(network: NN, x_train, y_train, x_test, y_test, lr=0.01, momentum=0.5, epochs=5, transform=(lambda x: x)):

    y_train = np.array(y_train).flatten()
    y_test = np.array(y_test).flatten()
    cost_per_epoch = Counter(range(1, epochs+1))
    for epoch in range(epochs):
        total_cost = 0
        indices = np.arange(len(x_train))
        np.random.shuffle(indices)
        x_train, y_train = x_train[indices], y_train[indices]
        for i in range(len(x_train)):
            network.learn(x_train[i], y_train[i], lr, momentum)
            total_cost += network._cost(x_train[i], y_train[i])

        print(f"Epoch {epoch+1}, Avg Cost: {total_cost / len(x_train)}")
        cost_per_epoch[epoch+1] = total_cost / len(x_train)

    print([cost_per_epoch[i] for i in range(1, epochs+1)])
    correct = 0
    accuracy = 0
    counter_size = np.max(y_train) + 1
    counts = Counter(range(counter_size))
    appearances = Counter(y_test)
    print([appearances[i] for i in range(counter_size)])

    flag = ""
    while not flag:
        flag = input('press key to continue... ')

    for i in range(len(x_test)):
        guess = network.classify(x_test[i])
        t_g = transform(guess)
        t_yt = transform(y_test[i])
        if guess == y_test[i]:
            counts[guess] += 1
            correct += 1
        else:
            print(f'Missed picture {i}, output {t_g} when answer was {t_yt}')
        accuracy = correct / (i + 1)

    print(f"Total Correct: {correct}, accuracy of {accuracy:.4f}")
    print([(transform(i), counts[i]) for i in range(counter_size)])
    print([(transform(i), appearances[i]) for i in range(counter_size)])


if not sys.argv[1] in ['digits', 'fashion', 'emnist']:
    exit()

if sys.argv[1] != 'none':
    dataset = sys.argv[1]
    x_train, y_train, x_test, y_test = read_data(dataset, noisy_train=False, noisy_test=False)
    layer_sizes = [x_train[0].shape[1], 256, 128, 64, np.max(y_train)+1]
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    n = NN(layer_sizes)
    train(n, x_train, y_train, x_test, y_test, lr=0.001, momentum=0.9, epochs=5)
    exit()
