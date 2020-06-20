from activation_function.Linear import Linear
from activation_function.Sigmoid import Sigmoid
from model.Network import Network

network = Network(2)
network.add_layer(3, Sigmoid)
network.add_layer(1, Sigmoid)

network.set_inputs([-2, 0])
network.set_desired_outputs([2.4])

network.set_all_weights_to(1)
network.forward_propagate()
print("outputs", network.get_outputs())

network.backward_propagate()
network.forward_propagate()

print("outputs", network.get_outputs())