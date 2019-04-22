import random
import numpy as np
from utils import *

class network(object):
    def __init__(self, hidden_units):
        self.num_hidden_units = hidden_units
        self.input_to_hidden_weights = self._init_hidden_layer_weights()
        self.hidden_to_output_weights = self._init_output_layer_weights()
        self.learning_rate = 0.1
        self.momentum = 0

    def sigmoid(self, z):
        return float(1)/(float(1)+np.exp(-z))

    def _init_hidden_layer_weights(self):
        #return np.random.random(0, 2, size=(self.num_hidden_units, 3), dtype=np.float32)
        return np.random.uniform(-0.05, 0.05, size=(self.num_hidden_units, 785))

    def _init_output_layer_weights(self):
        #return np.random.random(1, 2, size=(2, self.num_hidden_units+1), dtype=np.float32)
        return np.random.uniform(-0.05, 0.05, size=(10, (self.num_hidden_units+1)))

    def compute_hidden_layer(self, input_data):
        #print("Computing hidden layer")
        return np.dot(self.input_to_hidden_weights, input_data)

    def compute_output_layer(self, hidden_data):
        #print("Computing outer layer")
        return np.dot(self.hidden_to_output_weights, hidden_data)

    def output_error(self, t_values, output_values):
        #print("Output error")
        term2 = float(1) - output_values
        term3 = t_values - output_values
        return output_values*term2*term3

    def hidden_unit_error(self, hidden_values, output_error):
        #print("Hidden layer error")

        m, n = 10, self.num_hidden_units + 1
        weights = self.hidden_to_output_weights[np.arange(n) != np.array([0]*10)[:, None]].reshape(m, -1)

        term3 = np.sum(weights*output_error, axis=1)

        #print("term3: {}".format(term3))
        #print("hidden values: {}".format(hidden_values))
        #print("output error: {}".format(output_error))
        term2 = float(1) - hidden_values
        #print(term2.shape)
        return np.sum(hidden_values*term2*(term3[np.newaxis].T), axis=0)

    def update_hidden_output_weights(self, error, hidden_layer_output):
        #print("Update hidden layer to output layer")
        delta_w = hidden_layer_output*(self.learning_rate*error)
        self.hidden_to_output_weights = self.hidden_to_output_weights + delta_w

    def update_input_hidden_weights(self, error, x_data):
        #print("Update input to hidden layer")
        delta_w = x_data*(self.learning_rate*error)
        #print(delta_w.shape)
        self.input_to_hidden_weights = self.input_to_hidden_weights + delta_w.T


