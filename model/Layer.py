from model.Neuron import Neuron


class Layer:

    def __init__(self, n_inputs, n_neurons, f, is_input=False):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.is_input = is_input
        self.f = f
        self.neurons = [Neuron(self.n_inputs, f) for n in range(n_neurons)]
