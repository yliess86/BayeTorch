from torch import Tensor
from typing import Any
from typing import Tuple
from typing import Union

import torch.nn as nn


INT_2_TWO = Union[int, Tuple[int, int]]
INT_2_THREE = Union[int, Tuple[int, int, int]]
EPSILON = 1e-8


def int_2_two(n: INT_2_TWO) -> Tuple[int, int]:
    if isinstance(n, int):
        return (n, n)
    return n


def int_2_three(n: INT_2_THREE) -> Tuple[int, int, int]:
    if isinstance(n, int):
        return (n, n, n)
    return n


class BayesianModule(nn.Module):
    def __init__(self) -> None:
        super(BayesianModule, self).__init__()

    def reset(self) -> None:
        raise NotImplementedError("'reset' not implemented!")
        
    def reparametrize(self, mean: Tensor, std: Tensor) -> Tensor:
        eps = std.data.new(std.size()).normal_(mean=0.0, std=1.0)
        X =  mean + eps * std

        return X

    def froward(self, *args, **kwargs) -> Any:
        raise NotImplementedError("'forward' not implemented!")

    @property
    def kl_divergence(self) -> Tensor:
        raise NotImplementedError("'kl_divergence' not implemented!")

    def __str__(self) -> str:
        return "BaysianModule(default)"

    def __repr__(self) -> str:
        return str(self)