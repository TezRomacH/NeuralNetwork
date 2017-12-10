"""
Loss function
"""
import numpy as np

from tezromach.tensor import Tensor


class LossFunction:
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        raise NotImplementedError

    def __str__(self) -> str:
        return self.__class__.__name__


class MSE(LossFunction):
    """
    Mean squared error is:
    sum( (predicted - y)^2 )
    """

    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        return np.sum((predicted - actual) ** 2)

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return 2 * (predicted - actual)
