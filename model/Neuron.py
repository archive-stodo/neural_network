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
        self.desired_output = 0

    def compute_a(self):
        self.a = round(self.f.value(self.z), 6)

    def compute_z_derivative(self):
        return self.f.derivative(self.z)

    def set_error_term(self, error_term):
        self.error_term = error_term

    def set_all_weights_to(self, new_weight):
        self.weights = []
        [self.weights.append(new_weight) for i in range(self.n_inputs)]

    def clip_length_of_all_numbers(self):
        self.a = round(self.a, 6)
        self.z = round(self.z, 6)
        self.error_term = round(self.error_term, 6)
        self.bias = round(self.bias, 6)
        for weight_n in range(self.n_inputs):
            self.weights[weight_n] = round(self.weights[weight_n], 6)