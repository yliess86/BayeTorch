import torch
import torch.nn as nn


def init_bayesian_with_frequentist(
    bayesian: nn.Module,
    frequentist: nn.Module,
    freeze: bool = True
) -> nn.Module:
    bayesian_named_params = bayesian.named_parameters()
    frequentist_named_params = frequentist.named_parameters()

    filtr = lambda data: "log_alpha" not in data[0]

    bayesian_named_params = filter(filtr, bayesian_named_params)
    frequentist_named_params = filter(filtr, frequentist_named_params)

    named_params = zip(bayesian_named_params, frequentist_named_params)
    for (_, bayesian_param), (_, frequentist_param) in named_params:
        bayesian_param.data = frequentist_param.data
        bayesian_param.requires_grad = not freeze

    return bayesian