import random
import numpy as np

class network(object):
    def __init__(self, learning_rate, hidden_units):
        self.weights = None
        self.num_hidden_units = hidden_units
        self.learning_rate = float(learning_rate)

    def activation_function(self, z):
        out = float(1)/(float(1)+np.exp(-z))
        return out