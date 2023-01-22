import typing

import torch.nn

from nagl.config.model import ActivationFunction


class Sequential(torch.nn.Sequential):
    """A convenience wrapper around ``torch.nn.Sequential`` for constructing MLP modules
    with a specified number of linear and dropout layers combined with a specific
    activation function.
    """

    def __init__(
        self,
        in_feats: int,
        hidden_feats: typing.List[int],
        activation: typing.Optional[typing.List[torch.nn.Module]] = None,
        dropout: typing.Optional[typing.List[float]] = None,
    ):
        """
        Args:
            in_feats:  Number of input node features.
            hidden_feats: ``hidden_feats[i]`` gives the size of node representations
                after the i-th layer. ``len(hidden_feats)`` equals the number of layers.
            activation: ``activation[i]`` decides the activation function to apply to the
                i-th layer. ``len(activation)`` equals the number of layers. In no values
                are specified ReLU will be used after each layer.
            dropout: ``dropout[i]`` decides the dropout probability on the output of the
                i-th layer. ``len(dropout)`` equals the number of layers. If no values
                are specified then do dropout will take place.
        """

        n_layers = len(hidden_feats)

        # Initialize the default inputs.
        activation = [torch.nn.ReLU()] * n_layers if activation is None else activation
        dropout = [0.0] * n_layers if dropout is None else dropout

        # Validate that a consistent number of layers have been specified.
        lengths = [len(hidden_feats), len(activation), len(dropout)]

        if len({*lengths}) != 1:

            raise ValueError(
                "The `hidden_feats`, `activation`, and `dropout` lists must be the "
                "same length."
            )

        # Construct the sequential layer list.
        hidden_feats = [in_feats] + hidden_feats

        super().__init__(
            *(
                layer
                for i in range(n_layers)
                for layer in [
                    torch.nn.Linear(hidden_feats[i], hidden_feats[i + 1]),
                    activation[i],
                    torch.nn.Dropout(dropout[i]),
                ]
            )
        )


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
