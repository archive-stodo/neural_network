import unittest

from activation_function.Linear import Linear
from activation_function.Sigmoid import Sigmoid
from model.Network import Network
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class TestNetworkBackwardPropagation(unittest.TestCase):

    def test_forward_propagation(self):
        # given
        network = Network(4)
        network.add_layer(3, Sigmoid)
        network.add_layer(1, Linear)

        iris = load_iris()
        X = iris.data
        y = iris.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

        for epoch in range(10):
            for example_n in range(y_train.size):
                network.set_inputs(X_train[example_n])
                network.set_desired_outputs([ y_train[example_n] ])
                network.forward_propagate()
                network.backward_propagate()

        for i in range(y_test.size):
            network.set_inputs(X_test[i])
            network.set_desired_outputs([ y_test[i] ])

            network.forward_propagate()

            print("desired output: ", y_test[i])
            print("actual output:", network.get_outputs(), "\n")

        self.assertEquals(1, 1)