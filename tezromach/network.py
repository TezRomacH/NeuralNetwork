"""
A NeuralNet is just a collection of layers
"""
from typing import Sequence, Iterator, Tuple

import sys

from tezromach.iterators import DataIterator, BatchIterator
from tezromach.loss import Loss, MSE
from tezromach.tensor import Tensor
from tezromach.layers import Layer


class NeuralNet:
    def __init__(self, layers: Sequence[Layer]) -> None:
        self.layers = layers

    def predict(self, inputs: Tensor) -> Tensor:
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def _backward(self, grad: Tensor) -> Tensor:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def _params_and_grads(self) -> Iterator[Tuple[Tensor, Tensor]]:
        for layer in self.layers:
            for name, param in layer.params.items():
                grad = layer.grads[name]
                yield param, grad

    def _step(self, learning_rate: float) -> None:
        for param, grad in self._params_and_grads():
            param -= learning_rate * grad

    def _fit_layers(self, inputs: Tensor) -> None:
        for layer in self.layers:
            layer.fit(inputs)

    def __str__(self):
        return "NeuralNet, layers: [\n\t" + '\n\t'.join([str(layer) for layer in self.layers]) + "\n]"

    def fit(self,
            inputs: Tensor,
            targets: Tensor,
            learning_rate: float = 0.01,
            num_epochs: int = 1000,
            loss: Loss = MSE(),
            iterator: DataIterator = BatchIterator(),
            print_debug: bool = False) -> None:
        self._fit_layers(inputs)

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for batch in iterator(inputs, targets):
                predicted = self.predict(batch.inputs)
                epoch_loss += loss.loss(predicted, batch.targets)
                grad = loss.grad(predicted, batch.targets)
                self._backward(grad)
                self._step(learning_rate)
            if print_debug:
                print(epoch, epoch_loss, file=sys.stderr)
