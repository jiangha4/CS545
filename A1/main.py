import matplotlib
from utils import load_data, create_preceptron_layer, shuffle_data, append_bias


def compute(network, x_data):
    percept_output = []
    for percept in network:
        out = percept.compute_target(x_data)
        if out > 0:
            percept_output.append(1)
        else:
            percept_output.append(0)
    return percept_output


def update(network, t_values, percept_outputs, x_data):
    for i in range(0, len(network)):
        network[i].update_weights(t_values[i], percept_outputs[i], x_data)


def calc_acc(predictions, labels):
    correct = 0
    numOfPredictions = len(predictions)
    for i in range(0, numOfPredictions):
        if int(predictions[i]) == int(labels[i]):
            correct += 1
    acc = float(correct/numOfPredictions)
    print(acc)
    return acc

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

    # create 10 Perceptrons
    network = create_preceptron_layer(.001)

    # train
    acc = []
    for j in range(0, epochs):
        print("Training Epoch: {}".format(j))
        for i in range(0, x_train.shape[0]):
            out = compute(network, x_train[i])
            t_values = [0] * 10
            t_values[int(y_train[i])] = 1
            update(network, t_values, out, x_train[i])

        prediction = []
        for k in range(0, x_test.shape[0]):
            percept_output = compute(network, x_test[k])
            prediction.append(percept_output.index(max(percept_output)))

        acc.append(calc_acc(prediction, y_test))


if __name__ == '__main__':
    main()


