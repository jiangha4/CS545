__author__= "Haohan Jiang, 938737222"

import numpy as np
import os
import matplotlib.pyplot as plt

def generate_graph(acc_train, acc_test):
    numEpochs = list(range(0, 50))

    plt.plot(numEpochs, acc_train, label='Training Accuracy')
    plt.plot(numEpochs, acc_test, label='Testing Accuracy')

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    axes = plt.gca()
    axes.set_ylim([30, 100])

    plt.legend()
    plt.show()

def generate_matrix(predictions, y_labels):
    print("Generating confusion matrix")
    matrix = np.zeros([10, 10])
    for i in range(0, len(predictions)):
        matrix[int(y_labels[i])][int(predictions[i])] += 1

    print(matrix)

def calc_acc(predictions, labels):
    '''
    Calc accuracy
    '''
    correct = 0
    numOfPredictions = len(predictions)
    for i in range(0, numOfPredictions):
        if int(predictions[i]) == int(labels[i]):
            correct += 1
    acc = float(correct/numOfPredictions)
    return acc

def pickled_data():
    data_names = ['train', 'test']
    exists = True
    for name in data_names:
        path = os.path.dirname(os.path.abspath(__file__))
        file_path = str(path) + '/statics/' + name + '.npy'

        exists = exists & os.path.isfile(file_path)

    return exists

def save_data(test, train):
    print("Saving data to pickles")
    args = locals()
    data_names = ['test', 'train']
    for name in data_names:
        path = os.path.dirname(os.path.abspath(__file__))
        file_path = str(path) + '/statics/' + name
        np.save(file_path, args[name])

def load_saved_data():
    buf = {}
    data_names = ['train', 'test']
    for name in data_names:
        path = os.path.dirname(os.path.abspath(__file__))
        file_path = str(path) + '/statics/' + name + '.npy'

        buf[name] = np.load(file_path)
    return buf

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