__author__= "Haohan Jiang, 938737222"

from utils import *
import numpy as np

def get_data():
    if not pickled_data():
        # load training data
        print("Loading training data")
        path = os.path.dirname(os.path.abspath(__file__)) + '/datasets/mnist_train.csv'
        data = load_data(path)
        x_train, y_train = shuffle_data(data)

        # load testing data
        print("Loading testing data")
        path = os.path.dirname(os.path.abspath(__file__)) + '/datasets/mnist_test.csv'
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


def main():
    x_train, y_train, x_test, y_test = get_data()

if __name__ == "__main__":
    main()