"""
A NeuralNet is just a collection of layers
"""
from typing import Sequence, Iterator, Tuple, Optional

import sys
import pickle

from tezromach.iterators import DataIterator, BatchIterator
from tezromach.loss import LossFunction, MSE
from tezromach.tensor import Tensor
from tezromach.layers import Layer


class NeuralNet:
    def __init__(self, layers: Sequence[Layer]) -> None:
        self.layers: Sequence[Layer] = layers
        self._trained_params: Optional[str] = None

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

    def __str__(self) -> str:
        return "NeuralNet\n" + \
               (self._trained_params if self._trained_params else "") + \
               "layers: [\n\t" + '\n\t'.join([str(layer) for layer in self.layers]) + "\n]"

    def __repr__(self) -> str:
        return str(self)

    def fit(self,
            inputs: Tensor,
            targets: Tensor,
            learning_rate: float = 0.01,
            num_epochs: int = 1000,
            epsilon: Optional[float] = None,
            loss: LossFunction = MSE(),
            iterator: DataIterator = BatchIterator(),
            verbose_print: Optional[int] = None) -> None:
        self._fit_layers(inputs)
        self._trained_params = None
        count_epochs: int = 0
        epoch_loss = 0.0

        for epoch in range(1, num_epochs + 1):
            epoch_loss = 0.0
            for batch in iterator(inputs, targets):
                predicted = self.predict(batch.inputs)
                epoch_loss += loss.loss(predicted, batch.targets)
                grad = loss.grad(predicted, batch.targets)
                self._backward(grad)
                self._step(learning_rate)
            if verbose_print and epoch % verbose_print == 0:
                print("{}:\t{} = {}".format(epoch, loss, epoch_loss))
            if epsilon and epoch_loss <= epsilon:
                count_epochs = epoch
                break
        else:
            count_epochs = num_epochs

        self._trained_params = "trained_params:" \
                               "\n\tlearning_rate: {0}" \
                               "\n\tnum_epochs: {1}" \
                               "\n\tloss_function: {2} = {3}\n" \
            .format(learning_rate, count_epochs, str(loss), epoch_loss)

    def save_model(self, filename: str) -> None:
        with open(filename, 'wb') as f:
            pickle.dump(self, f)


def load_model(filename: str) -> NeuralNet:
    with open(filename, 'rb') as f:
        return pickle.load(f)
