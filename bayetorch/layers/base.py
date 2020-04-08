from torch import Tensor
from typing import Any
from typing import Tuple
from typing import Union

import torch.nn as nn


INT_2_TWO = Union[int, Tuple[int, int]]
EPSILON = 1e-10


def int_2_two(n: INT_2_TWO) -> Tuple[int, int]:
    if isinstance(n, int):
        return (n, n)
    return n


class BayesianModule(nn.Module):
    def __init__(self) -> None:
        super(BayesianModule, self).__init__()

    def reset(self) -> None:
        raise NotImplementedError("'reset' not implemented!")
        
    def reparametrize(self, mean: Tensor, std: Tensor) -> Tensor:
        eps = std.new(std.size()).normal_(0.0, 1.0)

        return std * eps + mean

    def froward(self, *args, **kwargs) -> Any:
        raise NotImplementedError("'forward' not implemented!")

    @property
    def kl_divergence(self) -> Tensor:
        raise NotImplementedError("'kl_divergence' not implemented!")

    def __str__(self) -> str:
        return "BaysianModule(default)"

    def __repr__(self) -> str:
        return str(self)