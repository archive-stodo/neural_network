from activation_function.Linear import Linear
from model.Layer import Layer
from model.Neuron import Neuron


class Network:

    def __init__(self, n_inputs):
        self.n_inputs = n_inputs
        self.layers = []
        self.add_input_layer(n_inputs)
        self.n_layers = 1
        self.last_layer = self.layers[self.n_layers - 1]

    def add_input_layer(self, n_inputs):
        input_layer = Layer(n_inputs, n_inputs, Linear, True)
        self.layers.append(input_layer)

    def add_layer(self, n_neurons, f):
        n_neurons_previous_layer = self.layers[-1].n_neurons
        new_layer = Layer(n_neurons_previous_layer, n_neurons, f, is_input=False)
        self.layers.append(new_layer)
        self.n_layers += 1
        self.last_layer = self.layers[self.n_layers - 1]


    def set_inputs(self, inputs):
        if len(inputs) != self.n_inputs:
            raise Exception('wrong input size')

        for neuron_n in range(len(inputs)):
            self.layers[0].neurons[neuron_n].a = inputs[neuron_n]

    def set_desired_outputs(self, outputs):
        if len(outputs) != self.last_layer.n_neurons:
            raise Exception('wrong output size')

        for neuron_n in range(len(outputs)):
            self.last_layer.neurons[neuron_n].desired_output = outputs[neuron_n]

    def get_outputs(self):
        outputs = [neuron.a for neuron in self.layers[self.n_layers - 1].neurons]
        return outputs

    def set_all_weights_to(self, new_weight):
        [layer.set_all_weights_to(new_weight) for layer in self.layers]

    def forward_propagate(self):
        for layer_n in range(1, self.n_layers):
            layer_neuron_n = self.layers[layer_n].n_neurons
            current_layer: Layer = self.layers[layer_n]
            previous_layer: Layer = self.layers[layer_n - 1]

            for neuron_n in range(layer_neuron_n):
                current_neuron: Neuron = current_layer.neurons[neuron_n]

                neuron_z = 0
                for previous_layer_neuron_n in range(previous_layer.n_neurons):
                    previous_layer_neuron_output = previous_layer.neurons[previous_layer_neuron_n].a
                    weight = current_neuron.weights[previous_layer_neuron_n]
                    neuron_z += previous_layer_neuron_output * weight

                current_neuron.z = neuron_z + current_neuron.bias
                current_neuron.compute_a()

    def backward_propagate(self):
        # last layer error term
        [neuron.set_error_term( (neuron.desired_output - neuron.a) * neuron.compute_a_derivative() ) for neuron in self.last_layer.neurons]

        # propagate error backwards
        for layer_n in range(self.n_layers - 2, 0, -1):
            print('processing layer n: ', layer_n)
            for neuron_n in range(self.layers[layer_n].n_neurons):
                neuron = self.layers[layer_n].neurons[neuron_n]
                error_term = 0
                for next_layer_neuron in self.layers[layer_n + 1].neurons:
                    error_term += next_layer_neuron.error_term * next_layer_neuron.weights[neuron_n] * neuron.compute_a_derivative()

                neuron.error_term = error_term

