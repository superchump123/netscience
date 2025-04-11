from data import read_data
from model import Model
from nn import NN

import numpy as np
import sys

bad_dataset = not sys.argv[1] in ['digits', 'fashion', 'emnist', 'diabetes']
mnist_like = ['digits', 'fashion', 'emnist']
if bad_dataset:
    exit()


def fashion_transform(x):
    global fashion_labels
    return fashion_labels[x]


def emnist_transform(x):
    return chr(x + ord('A'))


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
    case 'diabetes':
        def transform(x): return 'Yes' if x else 'No'
    case _:
        transform = None


dataset = sys.argv[1]
x_train, y_train, x_test, y_test = read_data(dataset, noisy_train=True, noisy_test=False)

if mnist_like:
    x_train = x_train / 255.0
    x_test = x_test / 255.0

match sys.argv[2]:
    case 'shallow':
        layer_sizes = [x_train[0].shape[1], 128, int(np.max(y_train)+1)]

    case 'deep':
        layer_sizes = [x_train[0].shape[1], 256, 128, 64, 32, int(np.max(y_train)+1)]

    case 'big':
        layer_sizes = [x_train[0].shape[1], 2 * (x_train[0].shape[1]), 128, 64, 32, int(np.max(y_train)+1)]

    case 'diabetes':
        layer_sizes = [x_train[0].shape[1], 10, int(np.max(y_train)+1)]

n = NN(layer_sizes)
model_name = f'{sys.argv[1]}_{sys.argv[2]}'
model = Model(n)
model.add_params(lr=0.001, momentum=0.9, layer_sizes=layer_sizes)
model.load_metadata(model_name)
loaded = model.load(model_name)
if not sys.argv[1]:
    exit()

if sys.argv[3] == 'train':
    model.train(x_train, y_train, 20)


elif sys.argv[3] == 'test':
    if transform:
        model.test(x_test, y_test, transform)
    else:
        model.test(x_test, y_test)

model.save_metadata(model_name)
model.save(model_name)
exit()
