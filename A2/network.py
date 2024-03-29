import random
import numpy as np
from utils import *

class network(object):
    def __init__(self, hidden_units, momentum):
        self.num_hidden_units = hidden_units
        self.input_to_hidden_weights = self._init_hidden_layer_weights()
        self.hidden_to_output_weights = self._init_output_layer_weights()
        self.learning_rate = 0.1
        self.momentum = momentum
        self.delta_w_output = np.zeros(self.hidden_to_output_weights.shape)
        self.delta_w_hidden = np.zeros(self.input_to_hidden_weights.shape)

    def sigmoid(self, z):
        return float(1)/(float(1)+np.exp(-z))

    def _init_hidden_layer_weights(self):
        return np.random.uniform(-0.05, 0.05, size=(self.num_hidden_units, 785))

    def _init_output_layer_weights(self):
        return np.random.uniform(-0.05, 0.05, size=(10, (self.num_hidden_units+1)))

    def compute_hidden_layer(self, input_data):
        return np.dot(self.input_to_hidden_weights, input_data)

    def compute_output_layer(self, hidden_data):
        return np.dot(self.hidden_to_output_weights, hidden_data)

    def output_error(self, t_values, output_values):
        term2 = float(1) - output_values
        term3 = t_values - output_values
        return output_values*term2*term3

    def hidden_unit_error(self, hidden_values, output_error):
        # reshape weights to exclude bias weight
        m, n = 10, self.num_hidden_units + 1
        weights = self.hidden_to_output_weights[np.arange(n) != np.array([0]*10)[:, None]].reshape(m, -1)

        term3 = np.dot(output_error, weights)
        term2 = float(1) - hidden_values
        return hidden_values*term2*term3

    def update_hidden_output_weights(self, error, hidden_layer_output):
        delta_w = hidden_layer_output*(self.learning_rate*error)
        self.hidden_to_output_weights = self.hidden_to_output_weights + delta_w\
                                        + (self.momentum*self.delta_w_output)
        self.delta_w_output = delta_w

    def update_input_hidden_weights(self, error, x_data):
        delta_w = x_data*(self.learning_rate*error)
        self.input_to_hidden_weights = self.input_to_hidden_weights + delta_w.T\
                                       + (self.momentum*self.delta_w_hidden)
        self.delta_w_hidden = delta_w.T


