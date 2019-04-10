__author__= "Haohan Jiang, 938737222"

import numpy as np

class Perceptron(object):
    def __init__(self, learning_rate):
        # 784 weights for input layer
        # 1 weight for bias
        self.weights = np.random.uniform(-0.5, 0.5, size=(785,))
        self.learning_rate = float(learning_rate)
        self.len = 785

    def compute_target(self, x_data):
        # compute the output for a single input
        return np.sum(np.dot(x_data, self.weights))

    def update_weights(self, t_value, previous_output, x_data):
        for i in range(0, self.len):
            w_delta = self.learning_rate*(t_value-previous_output)*x_data[i]
            self.weights[i] = self.weights[i] + w_delta

