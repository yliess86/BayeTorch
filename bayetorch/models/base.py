from bayetorch.convert import init_bayesian_with_frequentist
from torch import Tensor

import torch
import torch.nn as nn


class BayesianModel(nn.Module):
    def __init__(self) -> None:
        super(BayesianModel, self).__init__()

    @property
    def kl_divergence(self) -> Tensor:
        kld = torch.sum(torch.stack([
            child.kl_divergence 
            for child in self.children() 
            if hasattr(child, "kl_divergence")
        ]))
        return kld

    def init_with_frequentist(
        self,
        frequentist: nn.Module,
        freeze: bool = True
    ) -> "BayesianModel":
        return init_bayesian_with_frequentist(self, frequentist, freeze)