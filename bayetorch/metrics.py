from torch import Tensor

import torch
import torch.nn as nn
import torch.nn.functional as F


class ELBO(nn.Module):
    def __init__(self, n_samples: int) -> None:
        super(ELBO, self).__init__()
        self.n_samples = n_samples

    def forward(
        self,
        predictions: Tensor,
        targets: Tensor,
        kld: Tensor,
        kld_weight: float = 1.0
    ) -> Tensor:
        inputs = (predictions, targets)
        neg_log_likelihood = F.nll_loss(*inputs, reduction="mean")
        
        weighted_neg_log_likelihood = self.n_samples * neg_log_likelihood 
        weighted_kld = kld_weight * kld

        return weighted_neg_log_likelihood + weighted_kld

    @staticmethod
    def log_mean_exp(
        X: Tensor, 
        dim: int = None, 
        keepdim: bool = False
    ) -> Tensor:
        if dim is None:
            X, dim = X.view(-1), 0
        
        X_max, _ = torch.max(X, dim, keepdim=True)
        X_diff = X - X_max
        X = X_max + torch.log(torch.mean(torch.exp(X_diff), dim, keepdim=True))
        X = X if keepdim else X.squeeze(dim)
        
        return X