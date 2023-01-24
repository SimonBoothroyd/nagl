import typing

import torch.nn

from nagl.config.model import ActivationFunction


def get_activation_func(type_: ActivationFunction) -> typing.Type[torch.nn.Module]:
    """Return a PyTorch activation function of a given type.

    Args:
        type_: The type of activation function (e.g. 'ReLU').

    Returns:
        A function with signature ``(pred, target) -> metric``.
    """

    if type_.lower() == "identity":
        return torch.nn.Identity
    elif type_.lower() == "tanh":
        return torch.nn.Tanh
    elif type_.lower() == "relu":
        return torch.nn.ReLU
    elif type_.lower() == "leakyrelu":
        return torch.nn.LeakyReLU
    elif type_.lower() == "selu":
        return torch.nn.SELU
    elif type_.lower() == "elu":
        return torch.nn.ELU

    func = getattr(torch.nn, type_, None)

    if func is None:
        raise NotImplementedError(f"{type_} not a supported activation function")

    return func
