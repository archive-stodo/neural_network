from model.Neuron import Neuron


class Layer:

    def __init__(self, n_inputs, n_neurons, f):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.f = f
        self.neurons = [Neuron(self.n_inputs, f) for n in range(n_neurons)]

    def set_all_weights_to(self, new_weight):
        [neuron.set_all_weights_to(new_weight) for neuron in self.neurons]

    def clip_length_of_all_numbers(self):
        [neuron.clip_length_of_all_numbers() for neuron in self.neurons]
