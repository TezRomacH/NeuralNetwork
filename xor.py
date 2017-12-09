"""
The canonical example of a function that can't be
learned with a simple linear model is XOR
"""
import numpy as np

from tezromach.network import NeuralNet
from tezromach.layers import Linear, Sigmoid, Tanh, Normalization

inputs = np.array([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1]
])

targets = np.array([
    [0],
    [1],
    [1],
    [0]
])

net = NeuralNet([
    Normalization(),
    Linear(input_size=2, output_size=2),
    Sigmoid(),
    Linear(input_size=2, output_size=1),
    Tanh()
])

net.fit(inputs, targets, num_epochs=15000, learning_rate=0.02)

for x, y in zip(inputs, targets):
    predicted = net.predict(x)
    print(x, predicted, y)
