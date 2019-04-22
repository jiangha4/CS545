__author__= "Haohan Jiang, 938737222"

from utils import *
import numpy as np
from network import network


def get_data():
    if not pickled_data():
        # load training data
        print("Loading training data")
        path = os.path.dirname(os.path.abspath(__file__)) + '/datasets/mnist_train.csv'
        train_data = load_data(path)
        x_train, y_train = shuffle_data(train_data)

        # load testing data
        print("Loading testing data")
        path = os.path.dirname(os.path.abspath(__file__)) + '/datasets/mnist_test.csv'
        test_data = load_data(path)
        x_test, y_test = shuffle_data(test_data)

        # preprocessing
        x_train /= 255
        x_test /= 255

        # prepend bias
        print("Appending bias")
        x_train = append_bias(x_train)
        x_test = append_bias(x_test)

        save_data(train_data, test_data)
    else:
        buf = load_saved_data()
        train_data = buf['train']
        test_data = buf['test']

        x_train, y_train = shuffle_data(train_data)
        x_test, y_test = shuffle_data(test_data)

        # preprocessing
        x_train /= 255
        x_test /= 255

        # prepend bias
        x_train = append_bias(x_train)
        x_test = append_bias(x_test)

    return x_train, y_train, x_test, y_test

def dumby_data():
    x_test = x_train = np.array([[1, 0,0],[1, 1,0],[1, 0,1],[1, 1,1]])
    y_test = y_train = np.array([[0], [1], [1], [1]])

    return x_train, y_train, x_test, y_test

def data():
    if not pickled_data():
        print("Loading training data")
        test_data = load_data('datasets/mnist_test.csv')
        train_data = load_data('datasets/mnist_train.csv')
        save_data(test_data, train_data)
    else:
        buf = load_saved_data()
        train_data = buf['train']
        test_data = buf['test']

    return train_data, test_data

def main():
    x_train, y_train, x_test, y_test = get_data()
    #x_train, y_train, x_test, y_test = dumby_data()

    # epochs
    epochs = 50

    # number of hidden units
    hidden_units = [20, 50, 100]

    training_length = x_train.shape[0]
    test_length = x_test.shape[0]
    for num in hidden_units:
        neural_net = network(num, 0.9)

        acc_test_buf = []
        acc_train_buf = []
        for i in range(0, epochs):
            training_data, testing_data = data()
            x_train, y_train = shuffle_data(training_data)
            x_test, y_test = shuffle_data(testing_data)

            print(x_train.shape)
            x_train /= 255
            x_test /= 255

            x_train = append_bias(x_train)
            x_test = append_bias(x_test)

            print("Training Epoch: {}".format(i))
            prediction_training = []
            for j in range(0, training_length):
                x_2d = x_train[j][np.newaxis]
                # forward propagate
                # hidden layer
                hidden_layer_output = neural_net.compute_hidden_layer(x_2d.T)
                # apply activation
                output_layer_inputs = neural_net.sigmoid(hidden_layer_output.T)

                # output layer
                out_inputs = append_bias(output_layer_inputs).flatten()
                output_layer_outputs = neural_net.compute_output_layer(out_inputs.T)
                train_out_raw = neural_net.sigmoid(output_layer_outputs)

                # get prediction of outputs
                prediction_training.append(train_out_raw.argmax())

                # create target vector
                t_values = np.full((10,), 0.1)
                t_values[int(y_train[j])] = 0.9

                # error terms
                sigma_k = neural_net.output_error(t_values, train_out_raw.flatten())
                #print("Output error: {}".format(sigma_k))
                output_unit_error_2d = sigma_k[np.newaxis]
                sigma_j = neural_net.hidden_unit_error(output_layer_inputs, output_unit_error_2d.T)
                #print("hidden unit error: {}".format(sigma_j))
                #print(sigma_j.shape)

                # update weights
                #print("pre: {}".format(neural_net.hidden_to_output_weights))
                neural_net.update_hidden_output_weights(output_unit_error_2d.T, out_inputs)
                #print("post: {}".format(neural_net.hidden_to_output_weights))
                #print("pre: {}".format(neural_net.input_to_hidden_weights))
                neural_net.update_input_hidden_weights(sigma_j, x_2d.T)
                #print("post: {}".format(neural_net.input_to_hidden_weights))

            accuracy_training = calc_acc(prediction_training, y_train)*float(100)
            print("Training Accuracy: {}".format(accuracy_training))

            prediction_test = []
            # Testing
            for k in range(0, test_length):
                x_test_2d = x_test[k][np.newaxis]
                 # forward propagate
                # hidden layer
                hidden_layer_output = neural_net.compute_hidden_layer(x_test_2d.T)
                #print("hidden layer: {}".format(hidden_layer_output))
                # apply activation
                output_layer_inputs = neural_net.sigmoid(hidden_layer_output.T)

                # output layer
                out_inputs = append_bias(output_layer_inputs).flatten()
                output_layer_outputs = neural_net.compute_output_layer(out_inputs.T)
                train_out_raw = neural_net.sigmoid(output_layer_outputs)

                # get prediction of outputs
                prediction_test.append(train_out_raw.argmax())

            accuracy_test = calc_acc(prediction_test, y_test)*float(100)
            print("Testing Accuracy: {}".format(accuracy_test))

            acc_train_buf.append(accuracy_training)
            acc_test_buf.append(accuracy_test)

        # graph
        generate_graph(acc_train_buf, acc_test_buf)

        generate_matrix(prediction_test, y_test)

if __name__ == "__main__":
    main()