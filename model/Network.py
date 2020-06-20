from activation_function.Linear import Linear
from model.Layer import Layer


class Network:

    def __init__(self, n_inputs):
        self.n_inputs = n_inputs
        self.layers = []
        self.add_input_layer(n_inputs)

    def add_input_layer(self, n_inputs):
        input_layer = Layer(n_inputs, n_inputs, Linear, True)
        self.layers.append(input_layer)

    def add_layer(self, n_neurons, f):
        n_neurons_previous_layer = self.layers[-1].n_inputs
        new_layer = Layer(n_neurons_previous_layer, n_neurons, f, is_input=False)
        self.layers.append(new_layer)


    def set_inputs(self, inputs):
        if len(inputs) != self.n_inputs:
            raise Exception('wrong input size')

        for neuron_n in range(len(inputs)):
            self.layers[0].neurons[neuron_n] = inputs[neuron_n]