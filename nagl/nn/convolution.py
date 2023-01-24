import abc
import typing

import dgl
import dgl.nn.pytorch
import torch.nn
import torch.nn.functional
from typing_extensions import Literal

ActivationFunction = typing.Callable[[torch.tensor], torch.Tensor]

SAGEConvAggregatorType = Literal["mean", "gcn", "pool", "lstm"]

S = typing.TypeVar("S", bound=torch.nn.Module)
T = typing.TypeVar("T", bound=str)


class GCNStack(torch.nn.ModuleList, typing.Generic[S, T], abc.ABC):
    """A wrapper around a stack of GCN graph convolutional layers."""

    def __init__(
        self,
        in_feats: int,
        hidden_feats: typing.List[int],
        activation: typing.Optional[typing.List[torch.nn.Module]] = None,
        dropout: typing.Optional[typing.List[float]] = None,
        aggregator: typing.Optional[typing.List[T]] = None,
    ):
        """
        Args:
            in_feats: Number of input node features.
            hidden_feats: ``hidden_feats[i]`` gives the size of node representations
                after the i-th GCN layer. ``len(hidden_feats)`` equals the number of
                GCN layers.
            activation: ``activation[i]`` decides the activation function to apply to
                the i-th GCN layer. ``len(activation)`` equals the number of GCN layers.
                In no values are specified ReLU will be used after each layer.
            dropout: ``dropout[i]`` decides the dropout probability on the output of the
                i-th GCN layer. ``len(dropout)`` equals the number of GCN layers.
                By default, no dropout is performed for all layers.
            aggregator: ``aggregator[i]`` decides the aggregator type for the
                i-th GCN layer.
        """

        super(GCNStack, self).__init__()

        n_layers = len(hidden_feats)

        activation = [torch.nn.ReLU()] * n_layers if activation is None else activation
        dropout = [0.0] * n_layers if dropout is None else dropout

        default_aggregator = self._default_aggregator()
        aggregator = (
            [default_aggregator] * n_layers if aggregator is None else aggregator
        )

        lengths = [len(hidden_feats), len(activation), len(dropout), len(aggregator)]

        if len(set(lengths)) != 1:

            raise ValueError(
                f"`hidden_feats`, `activation`, `dropout` and `aggregator` must "
                f"be lists of the same length ({lengths})"
            )

        for i in range(n_layers):

            self.append(
                self._gcn_factory(
                    in_feats,
                    hidden_feats[i],
                    aggregator[i],
                    dropout[i],
                    activation[i],
                )
            )

            in_feats = hidden_feats[i]

    @classmethod
    @abc.abstractmethod
    def _default_aggregator(cls) -> T:
        """The default aggregator type to use for the GCN layers."""

    @classmethod
    @abc.abstractmethod
    def _gcn_factory(
        cls,
        in_feats: int,
        out_feats: int,
        aggregator_type: T,
        dropout: float,
        activation: ActivationFunction,
        **kwargs,
    ) -> S:
        """A function which returns an instantiated GCN layer.

        Args:
            in_feats: Number of input node features.
            out_feats: Number of output node features.
            activation: The activation function to.
            dropout: `The dropout probability.
            aggregator_type: The aggregator type, which can be one of ``"sum"``,
                ``"max"``, ``"mean"``.
            init_eps: The initial value of epsilon.
            learn_eps: If True epsilon will be a learnable parameter.

        Returns:
            The instantiated GCN layer.
        """

    def reset_parameters(self):
        """Reinitialize model parameters."""
        for gnn in self:
            gnn.reset_parameters()

    def forward(self, graph: dgl.DGLGraph, inputs: torch.Tensor) -> torch.Tensor:
        """Update node representations.

        Args:
            graph: The batch of graphs to operate on.
            inputs: The inputs to the layers with shape=(n_nodes, in_feats).

        Returns
            The output hidden features with shape=(n_nodes, hidden_feats[-1]).
        """
        for gnn in self:
            inputs = gnn(graph, inputs)

        return inputs


class SAGEConvStack(GCNStack[dgl.nn.pytorch.SAGEConv, SAGEConvAggregatorType]):
    """A wrapper around a stack of SAGEConv graph convolutional layers"""

    @classmethod
    def _default_aggregator(cls) -> SAGEConvAggregatorType:
        return "mean"

    @classmethod
    def _gcn_factory(
        cls,
        in_feats: int,
        out_feats: int,
        aggregator_type: SAGEConvAggregatorType,
        dropout: float,
        activation: ActivationFunction,
        **kwargs,
    ) -> dgl.nn.pytorch.SAGEConv:

        return dgl.nn.pytorch.SAGEConv(
            in_feats=in_feats,
            out_feats=out_feats,
            activation=activation,
            feat_drop=dropout,
            aggregator_type=aggregator_type,
        )


ConvolutionModule = typing.Union[GCNStack]


def get_convolution_layer(
    type_: typing.Literal["SAGEConv"],
) -> typing.Type[ConvolutionModule]:

    if type_.lower() == "sageconv":
        return SAGEConvStack

    raise NotImplementedError(f"{type_} not a supported convolution layer type")
