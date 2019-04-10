__author__= "Haohan Jiang, 938737222"
from utils import load_data, create_preceptron_layer, shuffle_data, append_bias
import numpy as np
import matplotlib.pyplot as plt

def compute(network, x_data):
    '''
    Compute the y of the network
    '''
    percept_output_raw = []
    percept_output_converted = []
    for percept in network:
        out = percept.compute_target(x_data)
        percept_output_raw.append(out)
        if out > 0:
            percept_output_converted.append(1)
        else:
            percept_output_converted.append(0)
    return percept_output_raw, percept_output_converted


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

    plt.legend()
    plt.show()

def generate_matrix(predictions, y_labels):
    print("Generating confusion matrix")
    matrix = np.zeros([10, 10])
    for i in range(0, len(predictions)):
        matrix[int(predictions[i])][int(y_labels[i])] += 1

    print(matrix)

def main():
    # load training data
    data = load_data('../datasets/mnist_train.csv')
    x_train, y_train = shuffle_data(data)

    # load testing data
    data = load_data('../datasets/mnist_test.csv')
    x_test, y_test = shuffle_data(data)

    # preprocessing
    x_train /= 255
    x_test /= 255

    x_train = append_bias(x_train)
    x_test = append_bias(x_test)

    # epochs
    epochs = 50

    learning_rates = [.001, .01, .1]
    for rate in learning_rates:
        print("Creating 10 Perceptrons with learning rate: {}".format(rate))
        # Create 10 Perceptrons with learning rate rate
        network = create_preceptron_layer(rate)

        acc_test_buf = []
        acc_train_buf = []
        for j in range(0, epochs):
            print("Training Epoch: {}".format(j))
            # Training
            prediction_training = []
            for i in range(0, x_train.shape[0]):
                train_out_raw, train_out_converted = compute(network, x_train[i])
                prediction_training.append(train_out_raw.index(max(train_out_raw)))
                t_values = [0] * 10
                t_values[int(y_train[i])] = 1
                update(network, t_values, train_out_converted, x_train[i])
            accuracy_training = calc_acc(prediction_training, y_train)*float(100)

            prediction_test = []
            # Testing
            for k in range(0, x_test.shape[0]):
                test_out_raw, test_out_converted = compute(network, x_test[k])
                prediction_test.append(test_out_raw.index(max(test_out_raw)))
            accuracy_test = calc_acc(prediction_test, y_test)*float(100)

            acc_train_buf.append(accuracy_training)
            acc_test_buf.append(accuracy_test)

        # Create the graph
        generate_graph(acc_train_buf, acc_test_buf)

        # Generate matrix
        generate_matrix(prediction_test, y_test)


if __name__ == '__main__':
    main()


