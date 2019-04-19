__author__= "Haohan Jiang, 938737222"
from utils import *
import numpy as np
import matplotlib.pyplot as plt
from preceptron import Perceptron


def compute(raw_output):
    if raw_output > 0:
        return 1
    else:
        return 0

def update(network, t_values, percept_outputs, x_data):
    '''
    Update the weights
    '''
    for i in range(0, len(network)):
        network[i].update_weights(t_values[i], percept_outputs[i], x_data)


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
    print("Training Accuracy: {}".format(acc))
    return acc

def generate_graph(acc_train, acc_test):
    numEpochs = list(range(0, 50))

    plt.plot(numEpochs, acc_train, label='Training Accuracy')
    plt.plot(numEpochs, acc_test, label='Testing Accuracy')

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    axes = plt.gca()
    axes.set_ylim([40, 100])

    plt.legend()
    plt.show()

def generate_matrix(predictions, y_labels):
    print("Generating confusion matrix")
    matrix = np.zeros([10, 10])
    for i in range(0, len(predictions)):
        matrix[int(predictions[i])][int(y_labels[i])] += 1

    print(matrix)

def dumby_data():
    x_train = np.random.randint(0, 5, size=(10, 5))
    x_test = np.random.randint(0, 5, size=(3, 5))

    y_train = np.random.randint(0, 1, size=(10, 1))
    y_test = np.random.randint(0, 1, size=(3, 1))

    return x_train, y_train, x_test, y_test

def main():
    x_train, y_train, x_test, y_test = get_data()
    #x_train, y_train, x_test, y_test = dumby_data()

    # epochs
    epochs = 50

    #vectorized compute function
    vc = np.vectorize(compute)

    learning_rates = [.001, .01, .1]

    # number of training objects
    training_length = x_train.shape[0]
    test_length = x_test.shape[0]
    for rate in learning_rates:
        print("Creating 10 Perceptrons with learning rate: {}".format(rate))
        # Create 10 Perceptrons with learning rate rate
        network = Perceptron(rate)

        acc_test_buf = []
        acc_train_buf = []
        for j in range(0, epochs):
            print("Training Epoch: {}".format(j))
            # Training
            prediction_training = []
            for i in range(0, training_length):
                x_2d = x_train[i][np.newaxis]

                train_out_raw = network.compute_target(x_2d.T)

                train_prediction = vc(train_out_raw)
                prediction_training.append(train_out_raw.argmax())

                t_values = np.zeros(10)
                t_values[int(y_train[i])] = 1

                network.update_weights(t_values, train_prediction, x_2d)
            accuracy_training = calc_acc(prediction_training, y_train)*float(100)

            prediction_test = []
            # Testing
            for k in range(0, test_length):
                x_test_2d = x_test[k][np.newaxis]

                test_out_raw = network.compute_target(x_test_2d.T)

                prediction_test.append(test_out_raw.argmax())
            accuracy_test = calc_acc(prediction_test, y_test)*100

            acc_train_buf.append(accuracy_training)
            acc_test_buf.append(accuracy_test)

        # Create the graph
        generate_graph(acc_train_buf, acc_test_buf)

        # Generate matrix
        generate_matrix(prediction_test, y_test)


if __name__ == '__main__':
    main()


