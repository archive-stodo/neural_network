from activation_function.Sigmoid import Sigmoid
from model.Network import Network

network = Network(2)
network.add_layer(3, Sigmoid)
network.add_layer(1, Sigmoid)

network.set_inputs([10, 10])

network.set_all_weights_to(1)

print("works")