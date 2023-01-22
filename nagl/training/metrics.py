import typing

import torch

from nagl.config.data import MetricType

MetricFunction = typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


mse = torch.nn.functional.mse_loss
mae = torch.nn.functional.l1_loss


def rmse(pred: torch.Tensor, label: torch.Tensor):
    return torch.sqrt(torch.nn.functional.mse_loss(pred, label))


def get_metric(type_: MetricType) -> MetricFunction:
    """Return a function that computes a given loss metric for a predicted and target
    value

    Args:
        type_: The type of metric.

    Returns:
        A function with signature ``(pred, target) -> metric``.
    """

    if type_.lower() == "rmse":
        return rmse
    elif type_.lower() == "mse":
        return mse
    elif type_.lower() == "mae":
        return mae

    raise NotImplementedError(f"{type_} not a supported loss metric")
