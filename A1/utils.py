__author__= "Haohan Jiang, 938737222 jiang4"

import numpy as np
import os

def pickled_data():
    data_names = ['x_test', 'x_train', 'y_test', 'y_train']
    exists = True
    for name in data_names:
        file_name = 'A2_' + name
        path = os.path.dirname(os.path.abspath(__file__))
        file_path = str(path) + '/statics/' + file_name + '.npy'

        exists = exists & os.path.isfile(file_path)

    return exists

def save_data(x_test, x_train, y_test, y_train):
    print("Saving data to pickles")
    args = locals()
    data_names = ['x_test', 'x_train', 'y_test', 'y_train']
    for name in data_names:
        file_name = 'A2_' + name
        path = os.path.dirname(os.path.abspath(__file__))
        file_path = str(path) + '/statics/' + file_name
        np.save(file_path, args[name])

def load_saved_data():
    buf = {}
    data_names = ['x_test', 'x_train', 'y_test', 'y_train']
    for name in data_names:
        file_name = 'A2_' + name
        path = os.path.dirname(os.path.abspath(__file__))
        file_path = str(path) + '/statics/' + file_name + '.npy'

        buf[name] = np.load(file_path)
    return buf

def get_data():
    if not pickled_data():
        # load training data
        print("Loading training data")
        path = 'datasets/mnist_train.csv'
        data = load_data(path)
        x_train, y_train = shuffle_data(data)

        # load testing data
        print("Loading testing data")
        path = 'datasets/mnist_test.csv'
        data = load_data(path)
        x_test, y_test = shuffle_data(data)

        # preprocessing
        x_train /= 255
        x_test /= 255

        # prepend bias
        print("Appending bias")
        x_train = append_bias(x_train)
        x_test = append_bias(x_test)

        save_data(x_test, x_train, y_test, y_train)
    else:
        print("Found pickled data")
        buf = load_saved_data()
        x_train = buf['x_train']
        x_test = buf['x_test']
        y_train = buf['y_train']
        y_test = buf['y_test']

    return x_train, y_train, x_test, y_test


def load_data(path):
    '''
    Opens a CSV file and returns a numpy array.

    :param path: Path to the CSV file to read
    :return: numpy array of the x and y data
    '''
    with open(path, 'r', newline='') as csvfile:
        data = np.loadtxt(csvfile, delimiter=',')
    return data

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