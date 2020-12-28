from typing import TYPE_CHECKING, List, Optional

import torch.nn
import torch.nn.functional
from pydantic import BaseModel, Field
from typing_extensions import Literal

if TYPE_CHECKING:
    ActivationFunction = str
else:
    ActivationFunction = Literal["Identity", "Tanh", "ReLU", "LeakyReLU", "ELU"]


class SequentialConfig(BaseModel):
    """Configuration options for a sequential set of dense NN layers."""

    in_feats: int = Field(..., description="The number of input node features.")

    hidden_feats: List[int] = Field(
        ...,
        description="``hidden_feats[i]`` gives the size of node representations after "
        "the i-th layer. ``len(hidden_feats)`` equals the number of layers.",
    )
    activation: Optional[List[ActivationFunction]] = Field(
        None,
        description="If not None, ``activation[i]`` gives the activation function to "
        "be used for the i-th layer. ``len(activation)`` equals the number of layers.",
    )
    dropout: Optional[List[float]] = Field(
        None,
        description="``dropout[i]`` decides the dropout probability on the output of "
        "the i-th layer. ``len(dropout)`` equals the number of layers.",
    )


class SequentialLayers(torch.nn.Module):
    """A convenience class for constructing a MLP model with a specified number
    of linear and dropout layers combined with a specific activation function.
    """

    def __init__(
        self,
        in_feats: int,
        hidden_feats: List[int],
        activation: Optional[List[torch.nn.Module]] = None,
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

        super().__init__()

        n_layers = len(hidden_feats)

        # Initialize the default inputs.
        if activation is None:
            activation = [torch.nn.ReLU()] * n_layers
        if dropout is None:
            dropout = [0.0] * n_layers

        # Validate that a consistent number of layers have been specified.
        lengths = [len(hidden_feats), len(activation), len(dropout)]

        if len({*lengths}) != 1:

            raise ValueError(
                "The `hidden_feats`, `activation`, and `dropout` lists must be the "
                "same length."
            )

        # Construct the sequential layer list.
        hidden_feats = [in_feats] + hidden_feats

        self.layers = torch.nn.Sequential(
            *(
                layer
                for i in range(n_layers)
                for layer in [
                    torch.nn.Linear(hidden_feats[i], hidden_feats[i + 1]),
                    activation[i],
                    torch.nn.Dropout(dropout[i]),
                ]
            ),
        )

    @classmethod
    def from_config(cls, config: SequentialConfig) -> "SequentialLayers":

        return cls(
            in_feats=config.in_feats,
            hidden_feats=config.hidden_feats,
            activation=(
                None
                if config.activation is None
                else [
                    getattr(torch.nn, activation)() for activation in config.activation
                ]
            ),
            dropout=config.dropout,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)
