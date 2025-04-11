from collections import Counter
import numpy as np
from nn import NN
import json
import os


class Model:

    def __init__(self, nn: NN):
        self.nn = nn
        self.metadata = None

    def add_params(self, **kwargs):
        if not self.metadata:
            self.metadata = {}
        for k, v in kwargs.items():
            self.metadata[k] = v

    def load(self, filename, path='models/'):
        md = self.metadata['layer_sizes']
        nn = self.nn.layer_sizes
        if len(nn) != len(md):
            print('Cannot load model: layer amounts do not match.')
            return False
        for i in range(len(nn)):
            if int(nn[i]) != int(md[i]):
                print('Cannot load model: layer sizes do not match.')
                return False

        self.nn.load(os.path.join(path, f'{filename}.npz'))
        return True

    def load_metadata(self, filename, path='models/'):
        with open(os.path.join(path, 'models.json'), 'r') as f:
            self.metadata = json.load(f).get(filename, self.metadata)

    def save(self, filename, path='models/'):
        self.nn.save(os.path.join(path, f'{filename}.npz'))

    def save_metadata(self, filename, path='models/'):
        with open(os.path.join(path, 'models.json'), 'r+') as f:
            try:
                data = json.load(f)
            except:
                data = {}
            data[filename] = self.metadata
            f.seek(0)
            json.dump(data, f, indent=4)

    def train(self, x_train, y_train, epochs):
        y_train = np.array(y_train).flatten()
        cost_per_epoch = Counter(range(1, epochs+1))
        for epoch in range(epochs):
            total_cost = 0
            indices = np.arange(len(x_train))
            np.random.shuffle(indices)
            x_train, y_train = x_train[indices], y_train[indices]
            for i in range(len(x_train)):
                total_cost += self.nn.learn(x_train[i], y_train[i], self.metadata['lr'], self.metadata['momentum'])

            cost_per_epoch[epoch+1] = total_cost / len(x_train)
            print(f'Finished epoch {epoch + 1}')
        print([f'{cost_per_epoch[i]:.4f}' for i in range(1, epochs+1)])
    
    def test(self, x_test, y_test, transform=(lambda x: x)):
        y_test = np.array(y_test).flatten()
        correct = 0
        accuracy = 0
        counter_size = np.max(y_test) + 1
        counts = Counter(range(counter_size))
        misses = Counter(range(counter_size))
        appearances = Counter(y_test)
        print([appearances[i] for i in range(counter_size)])
        for i in range(len(x_test)):
            guess = self.nn.classify(x_test[i])
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
