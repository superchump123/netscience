import numpy as np
import polars as pl
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


def read_diabetes_data():
    df = pl.read_csv('data/diabetes/diabetes_dataset.csv')

    df = df.with_columns(
        (pl.col("Fasting_Blood_Glucose") > 125).cast(pl.Int8).alias("Outcome")
    )
    df = df.drop("")
    labels = df["Outcome"]
    features = df.drop("Outcome")

    nary_cols = ['Smoking_Status', 'Alcohol_Consumption', 'Physical_Activity_Level', 'Ethnicity', 'Sex']

    features = features.to_dummies(columns=nary_cols)

    features = features.with_columns([
        ((pl.col(c) - pl.col(c).mean()) / pl.col(c).std()).alias(c) for c in features.columns
    ])

    x = features.to_numpy().astype(np.float32)
    y = labels.to_numpy().astype(int)

    test_percent = .2
    test_start_x = int(len(x)*(1-test_percent))
    test_start_y = int(len(y)*(1-test_percent))
    return np.matrix(x[:test_start_x]), np.matrix(y[:test_start_y]), np.matrix(x[test_start_x:]), np.matrix(y[test_start_y:])


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

    def load_data(self, noisy_train=True, noisy_test=False):
        x_train, y_train = self.read_images_labels(
            self.training_images_filepath, self.training_labels_filepath, noisy_train)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath, noisy_test)
        return (x_train, y_train), (x_test, y_test)


def read_data(dataset, noisy_train=True, noisy_test=False):
    if dataset == 'diabetes':
        noisy_train = False
        noisy_test = False
        return read_diabetes_data()

    elif dataset == 'digits':
        training_images_filepath = join('data', dataset, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
        training_labels_filepath = join('data', dataset, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
        test_images_filepath = join('data', dataset, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
        test_labels_filepath = join('data', dataset, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')
    elif dataset == 'fashion':
        training_images_filepath = join('data', dataset, 'train-images-idx3-ubyte')
        training_labels_filepath = join('data', dataset, 'train-labels-idx1-ubyte')
        test_images_filepath = join('data', dataset, 't10k-images-idx3-ubyte')
        test_labels_filepath = join('data', dataset, 't10k-labels-idx1-ubyte')
    elif dataset == 'emnist':
        training_images_filepath = join('data', dataset, 'emnist_source_files',
                                        'emnist-letters-train-images-idx3-ubyte')
        training_labels_filepath = join('data', dataset, 'emnist_source_files',
                                        'emnist-letters-train-labels-idx1-ubyte')
        test_images_filepath = join('data', dataset, 'emnist_source_files', 'emnist-letters-test-images-idx3-ubyte')
        test_labels_filepath = join('data', dataset, 'emnist_source_files', 'emnist-letters-test-labels-idx1-ubyte')

    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath,
                                       test_images_filepath, test_labels_filepath)

    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data(noisy_train, noisy_test)

    if dataset == 'emnist':
        y_train = y_train - 1
        y_test = y_test - 1

    return x_train, y_train, x_test, y_test
