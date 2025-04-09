import numpy as np
import pandas as pd
from scipy.ndimage import rotate
import struct
from array import array
from os.path import join


def add_noise(image, noise_level=0.2):
    noise = np.random.normal(loc=0.0, scale=noise_level, size=image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 255)


def random_rotation(image, max_angle=15):
    angle = np.random.uniform(-max_angle, max_angle)
    return rotate(image, angle, reshape=False, mode='nearest')


class MnistDataloader(object):
    def __init__(self, training_images_filepath, training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(self, images_filepath, labels_filepath, noise):
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = np.array(array("B", file.read()))

        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())

        images = []
        for i in range(size):
            images.append(np.zeros(rows * cols))
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            images[i] = img

        if noise:
            images = [img.reshape((28, 28)) for img in images]

            # Augment
            augmented_images = []
            augmented_labels = []
            for img, label in zip(images, labels):
                augmented_images.append(random_rotation(img))
                augmented_images.append(add_noise(img))
                augmented_labels.append(label)
                augmented_labels.append(label)

            # Merge
            images.extend(augmented_images)
            labels = np.concatenate((labels, augmented_labels))
            images = [img.flatten() for img in images]

        return np.matrix(images), np.matrix(labels)

    def load_data(self, noisy_test=False):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath, True)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath, noisy_test)
        return (x_train, y_train), (x_test, y_test)


def read_data(input_path, noisy_test=False):
    training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
    training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
    test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
    test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath,
                                       test_images_filepath, test_labels_filepath)

    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data(noisy_test)
    return x_train, y_train, x_test, y_test
