from typing import TYPE_CHECKING, List, Optional

import torch.nn
import torch.nn.functional
from typing_extensions import Literal

if TYPE_CHECKING:
    ActivationFunction = str
else:
    ActivationFunction = Literal["Identity", "Tanh", "ReLU", "LeakyReLU", "ELU"]


class SequentialLayers(torch.nn.Sequential):
    """A convenience class for constructing a MLP model with a specified number
    of linear and dropout layers combined with a specific activation function.
    """

    def __init__(
        self,
        in_feats: int,
        hidden_feats: List[int],
        activation: Optional[List[ActivationFunction]] = None,
        dropout: Optional[List[float]] = None,
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
        activation = (
            [torch.nn.ReLU()] * n_layers
            if activation is None
            else [getattr(torch.nn, name)() for name in activation]
        )
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
