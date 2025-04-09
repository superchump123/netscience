from collections import Counter
from data import read_data
from nn import NN

import numpy as np
import sys


def train(network: NN, x_train, y_train, x_test, y_test, lr=0.01, momentum=0.5, epochs=5, report_every=1000):

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

            if (i + 1) % report_every == 0 or i == len(x_train) - 1:
                avg_so_far = total_cost / (i + 1)
                print(f"  Progress: {i+1}/{len(x_train)} — Avg Cost: {avg_so_far:.4f}")

        print(f"Epoch {epoch+1}, Avg Cost: {total_cost / len(x_train)}")
        cost_per_epoch[epoch+1] = total_cost / len(x_train)

    print([cost_per_epoch[i] for i in range(1, epochs+1)])
    correct = 0
    accuracy = 0
    counter_size = np.max(y_train)
    counts = Counter(range(counter_size))
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
            print(f'Missed picture {i}, output {guess + ord("A")} when answer was {y_test[i] + ord("A")}')
        accuracy = correct / (i + 1)
        if (i + 1) % report_every == 0 or i == len(x_test) - 1:
            print(f"Current accuracy: {accuracy:.4f}")
    print(f"Total Correct: {correct}, accuracy of {accuracy:.4f}")
    print([counts[i] for i in range(10)])
    print([appearances[i] for i in range(10)])


if not sys.argv[1] in ['digits', 'fashion', 'dataset']:
    exit()

if sys.argv[1] != 'none':
    dataset = sys.argv[1]
    x_train, y_train, x_test, y_test = read_data(dataset, noisy_train=True, noisy_test=False)
    layer_sizes = [x_train[0].shape[1], 128, 64, 26]
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    n = NN(layer_sizes)
    train(n, x_train, y_train, x_test, y_test, lr=0.001, momentum=0.9, epochs=10)
    exit()

