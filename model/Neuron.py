import numpy as np

class Neuron:

    def __init__(self, n_inputs, f):
        self.f = f
        self.n_inputs = n_inputs
        self.weights = np.array(np.random.randn(n_inputs)) / 10
        self.bias = 1
        self.z = 0
        self.a = 0
        self.error_term = 0

    def compute_a(self):
        self.a = self.f.value(self.z)
