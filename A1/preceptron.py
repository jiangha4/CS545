__author__= "Haohan Jiang, 938737222"

import numpy as np

class Perceptron(object):
    def __init__(self, learning_rate):
        # 784 weights for input layer
        # 1 weight for bias
        self.weights = np.random.uniform(-0.05, 0.06, size=(10, 785))
        self.learning_rate = float(learning_rate)
        self.len = 785

    def compute_target(self, x_data):
        return np.dot(self.weights, x_data)

    def update_weights(self, t_value, previous_output, x_data):
        learning_decision = t_value[np.newaxis].T - previous_output
        delta_w = np.matmul((self.learning_rate*learning_decision), x_data)
        self.weights = self.weights + delta_w





