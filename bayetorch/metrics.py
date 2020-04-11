from bayetorch.models.base import BayesianModel
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


class Uncertainty:
    def __init__(self, samples: int) -> None:
        self.samples = samples

    def __call__(self, model: BayesianModel, X: Tensor) -> None:
        with torch.no_grad():
            X = torch.cat(self.samples * [X.unsqueeze(0)])
            y, _ = model(X)
            y = F.softplus(y)

            p_hat = y / torch.sum(y, dim=1, keepdim=True)
            p_bar = torch.mean(p_hat, dim=0)

            epsitemic = p_hat - p_bar.unsqueeze(0)
            epsitemic = (epsitemic.T @ epsitemic) / self.samples

            aleatoric = torch.diag(p_bar)
            aleatoric = aleatoric - ((p_hat.T @ p_hat) / self.samples)

            prediction = torch.sum(y, dim=0).squeeze(0)
            prediction = torch.argmax(prediction)

        return epsitemic, aleatoric, prediction.item()