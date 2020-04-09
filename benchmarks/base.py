from typing import List
from typing import Tuple
from torch import Tensor

import numpy as np
import torch.nn as nn


class Benchmark:
    def run_frequentist(self) -> Tuple[float, float]:
        raise NotImplementedError("'run_frequentist' not implemented.")

    def run_bayesian(self) -> Tuple[float, float]:
        raise NotImplementedError("'run_bayesian' not implemented.")

    def __call__(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        frequentist = self.run_frequentist()
        bayesian = self.run_bayesian()

        return frequentist, bayesian


class SizeEstimator(object):
    def __init__(
        self,
        model: nn.Module,
        input_size: List[int],
        bit_size: int = 32
    ) -> None:
        self.model = model
        self.input_size = input_size
        self.bit_size = 32

    @property
    def parameter_sizes(self) -> List[np.ndarray]:
        return [
            np.array(parameter.size())
            for module in self.model.modules()
            for parameter in module.parameters()
        ]

    @property
    def param_bits(self) -> int:
        parameter_sizes = self.parameter_sizes
        bits = np.sum([
            np.prod(size) for size in parameter_sizes]
        ) * self.bit_size
        
        return bits

    def __call__(self) -> Tuple[float, int]:
        bits = self.param_bits
        megabytes = (bits / 8) / (1024 ** 2)
        
        return megabytes, bits