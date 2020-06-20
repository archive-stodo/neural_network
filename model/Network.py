from activation_function.Linear import Linear
from model.Layer import Layer
from model.Neuron import Neuron


class Network:

    def __init__(self, n_inputs):
        self.n_inputs = n_inputs
        self.layers = []
        self.add_input_layer(n_inputs)
        self.n_layers = 1

    def add_input_layer(self, n_inputs):
        input_layer = Layer(n_inputs, n_inputs, Linear, True)
        self.layers.append(input_layer)

    def add_layer(self, n_neurons, f):
        n_neurons_previous_layer = self.layers[-1].n_inputs
        new_layer = Layer(n_neurons_previous_layer, n_neurons, f, is_input=False)
        self.layers.append(new_layer)
        self.n_layers += 1


    def set_inputs(self, inputs):
        if len(inputs) != self.n_inputs:
            raise Exception('wrong input size')

        for neuron_n in range(len(inputs)):
            self.layers[0].neurons[neuron_n].a = inputs[neuron_n]

    def forward_propagate(self):
        for layer_n in range(1, self.n_layers):
            layer_neuron_n = len(self[layer_n].neurons)
            current_layer: Layer = self[layer_n]
            previous_layer: Layer = self[layer_n - 1]

            for neuron_n in range(layer_neuron_n):
                current_neuron: Neuron = current_layer.neurons[neuron_n]

                neuron_z = 0
                for previous_layer_neuron_n in range(previous_layer.n_neurons):
                    previous_layer_neuron_output = previous_layer.neurons[previous_layer_neuron_n].a
                    weight = current_neuron.weights[previous_layer_neuron_n]
                    neuron_z += previous_layer_neuron_output * weight

                current_neuron.z = neuron_z + current_neuron.bias
                current_neuron.compute_a()