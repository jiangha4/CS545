__author__= "Haohan Jiang, 938737222"

import numpy as np
from preceptron import Perceptron


def load_data(path):
    '''
    Opens a CSV file and returns a numpy array.

    :param path: Path to the CSV file to read
    :return: numpy array of the x and y data
    '''
    with open(path, 'r', newline='') as csvfile:
        data = np.loadtxt(csvfile, delimiter=',')
    return data


def create_preceptron_layer(learning_rate):
    '''
    Creates an array of 10 preceptrons with a given learning rate
    :param learning_rate: Learning rate of the preceptron
    :return: array of Preceptron objects
    '''
    preceptron_layer = []
    for i in range(0, 10):
        preceptron_layer.append(Perceptron(learning_rate))

    return preceptron_layer


def shuffle_data(data):
    '''
    Shuffles the data in place and returns data split by inputs and label
    :param data: numpy array of all data read from CSV file
    :return: x_data: numpy array of input data
             y_data: numpy array of label data
    '''
    np.random.shuffle(data)

    y_data = data[:, [0]]
    x_data = data[:, np.r_[1:785]]

    return x_data.astype('float32'), y_data


def append_bias(x_data):
    '''
    Prepends the bias input to the x_data
    :param x_data: numpy of input data
    :return: numpy array of appended bias and data
    '''

    num = x_data.shape[0]
    bias = np.ones((num, 1))

    return np.concatenate((bias, x_data), axis=1)