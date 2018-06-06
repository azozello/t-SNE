import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import struct


label_index = []
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']


def read_MNIST(filename_images, filename_labels):
    """
    Return MNIST data set from files in IDX format.
    """
    pixels = _read_images(filename_images)
    labels = _read_labels(filename_labels)

    # print_data_2d(pixels, labels)

    return [pixels, labels]


def _read_images(filename):
    """
    Read images in MNIST data set from file in IDX format.
    """
    with open(filename, 'rb') as fin:
        fin.seek(4)  # skip "magic number"
        n_samples = struct.unpack('>i', fin.read(4))[0]
        n_rows = struct.unpack('>i', fin.read(4))[0]
        n_cols = struct.unpack('>i', fin.read(4))[0]
        n_dim = n_cols * n_rows

        # data starts from byte 16
        pixels = np.fromfile(fin, dtype=np.ubyte)

    pixels = pixels.reshape(n_samples, n_dim)
    return pixels
    # return get_3d(pixels)


def get_2d(pixels):
    """
    Create 2D data-set
    """
    result = [[pixels[i][256], pixels[i][257]] for i in range(len(pixels))]
    test = []
    for i in range(len(result)):
            if result[i][0] > 0 and result[i][1] > 0:
                test.append([result[i][0], result[i][1]])
                label_index.append(i)

    return np.array(test)


def print_data_2d(pixels, labels):
    for i in range(len(pixels)):
        index = labels[i]
        if index > 7:
            index = index - 8
        plt.plot(pixels[i][0], pixels[i][1], colors[index] + '+')

    plt.title('2D images')
    plt.show()
    plt.clf()


def create_line():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    line = [[i, i, i] for i in range(1882)]

    for i in range(len(line)):
        ax.scatter(line[i][0], line[i][1], line[i][2], c='r', marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()
    plt.clf()

    return np.array(line)


def get_3d(pixels):
    """
    Create 3D data-set
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    result = [[pixels[i][256], pixels[i][257], pixels[i][258]] for i in range(len(pixels))]
    test = []
    for i in range(len(result)):
            if result[i][0] > 0 and result[i][1] > 0 and result[i][2] > 0:
                test.append([result[i][0], result[i][1], result[i][2]])
                label_index.append(i)
                ax.scatter(result[i][0], result[i][1], result[i][2], c='r', marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()
    plt.clf()
    return np.array(test)


def _read_labels(filename):
    """
    Read labels in MNIST data set from file in IDX format.
    """
    with open(filename, 'rb') as fin:
        fin.seek(8)  # skip "magic number" and number of labels
        # data starts from byte 8
        labels = np.fromfile(fin, dtype=np.ubyte)

    return labels
    # return np.array([labels[label_index[i]] for i in range(len(label_index))])
