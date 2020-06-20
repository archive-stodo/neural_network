import numpy as np

class Sigmoid:

    @classmethod
    def value(cls, input_x):
        return 1 / (1 + np.exp(-input_x))

    @classmethod
    def derivative(cls, input_x):
        return cls.value(input_x) * (1 - cls.value(input_x))