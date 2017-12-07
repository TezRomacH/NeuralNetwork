from typing import Sequence
import numpy as np

from tezromach.layers import Activation, Linear, Sigmoid, Tanh
from tezromach.network import NeuralNet


def create_nn_one_hidden(
        input_size: int,
        output_size: int,
        activation: Activation = Sigmoid()
) -> NeuralNet:
    if input_size < 0 or output_size < 0:
        raise Exception

    return NeuralNet([
        Linear(input_size=input_size, output_size=input_size),
        activation,
        Linear(input_size=input_size, output_size=output_size),
        activation
    ])


def create_nn(
        neurons_in_layers: Sequence[int],
        activation: Activation = Sigmoid()
) -> NeuralNet:
    n_layers = len(neurons_in_layers)
    layers = np.array([(Linear(input_size=neurons_in_layers[i], output_size=neurons_in_layers[i+1]), activation) for i in range(n_layers - 1)]).flatten()
    return NeuralNet(layers)
