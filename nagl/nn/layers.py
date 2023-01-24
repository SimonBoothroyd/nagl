import typing

import torch


class Sequential(torch.nn.Sequential):
    """A convenience wrapper around ``torch.nn.Sequential`` for constructing MLP nn
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
            activation: ``activation[i]`` decides the activation function to apply to
                the i-th layer. ``len(activation)`` equals the number of layers. In no
                values are specified ReLU will be used after each layer.
            dropout: ``dropout[i]`` decides the dropout probability on the output of the
                i-th layer. ``len(dropout)`` equals the number of layers. If no values
                are specified then no dropout will take place.
        """

        n_layers = len(hidden_feats)

        activation = [torch.nn.ReLU()] * n_layers if activation is None else activation
        dropout = [0.0] * n_layers if dropout is None else dropout

        lengths = [len(hidden_feats), len(activation), len(dropout)]

        if len({*lengths}) != 1:

            raise ValueError(
                f"`hidden_feats`, `activation`, and `dropout` must be lists of the "
                f"same length ({lengths})"
            )

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
