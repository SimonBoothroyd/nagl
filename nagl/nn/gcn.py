# The ``GraphGCNStack`` class is based upon the ``dgllife.model.GraphSAGE`` module which
# is licensed under:
#
#     Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#     SPDX-License-Identifier: Apache-2.0
#
# The code has been modified so that the architecture of the GCN is abstracted.
import abc
from typing import Callable, Generic, List, Optional, TypeVar

import dgl
import dgl.nn.pytorch
import torch.nn
import torch.nn.functional
from typing_extensions import Literal

ActivationFunction = Callable[[torch.tensor], torch.Tensor]

SAGEConvAggregatorType = Literal["mean", "gcn", "pool", "lstm"]
GINConvAggregatorType = Literal["sum", "max", "mean"]

S = TypeVar("S", bound=torch.nn.Module)
T = TypeVar("T", bound=str)


class GCNStack(torch.nn.Module, Generic[S, T], abc.ABC):
    """A wrapper around a stack of GCN graph convolutional layers.

    Note:
        This class is based on the ``dgllife.model.SAGEConv`` module.
    """

    def __init__(
        self,
        in_feats: int,
        hidden_feats: List[int],
        activation: Optional[List[ActivationFunction]] = None,
        dropout: Optional[List[float]] = None,
        aggregator_type: Optional[List[T]] = None,
    ):
        """
        Args:
            in_feats: Number of input node features.
            hidden_feats: ``hidden_feats[i]`` gives the size of node representations
                after the i-th GCN layer. ``len(hidden_feats)`` equals the number of
                GCN layers.
            activation: If not None, ``activation[i]`` gives the activation function to
                be used for the i-th GCN layer. ``len(activation)`` equals the number
                of GCN layers.
            dropout: ``dropout[i]`` decides the dropout probability on the output of the
                i-th GCN layer. ``len(dropout)`` equals the number of GCN layers.
                By default, no dropout is performed for all layers.
            aggregator_type: ``aggregator_type[i]`` decides the aggregator type for the
                i-th GCN layer.
        """

        super(GCNStack, self).__init__()

        n_layers = len(hidden_feats)

        # Set the default options.
        if activation is None:
            activation = [torch.nn.functional.relu for _ in range(n_layers)]
        if dropout is None:
            dropout = [0.0 for _ in range(n_layers)]
        if aggregator_type is None:
            aggregator_type = [self._default_aggregator_type() for _ in range(n_layers)]

        lengths = [
            len(hidden_feats),
            len(activation),
            len(dropout),
            len(aggregator_type),
        ]

        if len(set(lengths)) != 1:

            raise ValueError(
                f"`hidden_feats`, `activation`, `dropout` and `aggregator_type` must "
                f"be lists of the same length ({lengths})"
            )

        self.hidden_feats = hidden_feats
        self.gnn_layers = torch.nn.ModuleList()

        for i in range(n_layers):

            self.gnn_layers.append(
                self._gcn_factory(
                    in_feats,
                    hidden_feats[i],
                    aggregator_type[i],
                    dropout[i],
                    activation[i],
                )
            )

            in_feats = hidden_feats[i]

    @classmethod
    @abc.abstractmethod
    def _default_aggregator_type(cls) -> T:
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
        for gnn in self.gnn_layers:
            gnn.reset_parameters()

    def forward(self, graph: dgl.DGLGraph, inputs: torch.Tensor) -> torch.Tensor:
        """Update node representations.

        Args:
            graph: The batch of graphs to operate on.
            inputs: The inputs to the layers with shape=(n_nodes, in_feats).

        Returns
            The output hidden features with shape=(n_nodes, hidden_feats[-1]).
        """
        for gnn in self.gnn_layers:
            inputs = gnn(graph, inputs)

        return inputs


#
# TODO: This does not seem to work...
#
# class GINConv(torch.nn.Module):
#     """Represents a single GINConv layer which is applied according to
#     ``activation(gin(dropout(x)))``.
#
#     The order of these operations was chosen to match the `dgl.nn.SAGEConv` module.
#     """
#
#     def __init__(
#         self,
#         in_feats: int,
#         out_feats: int,
#         activation: ActivationFunction,
#         dropout: float = 0.0,
#         aggregator_type: GINConvAggregatorType = "sum",
#         init_eps: float = 0.0,
#         learn_eps: bool = False,
#     ):
#         """
#
#         Args:
#             in_feats: Number of input node features.
#             out_feats: Number of output node features.
#             activation: The activation function to.
#             dropout: `The dropout probability.
#             aggregator_type: The aggregator type, which can be one of ``"sum"``,
#                 ``"max"``, ``"mean"``.
#             init_eps: The initial value of epsilon.
#             learn_eps: If True epsilon will be a learnable parameter.
#         """
#
#         super(GINConv, self).__init__()
#
#         self.activation = activation
#         self.dropout = torch.nn.Dropout(dropout)
#         self.gcn = dgl.nn.pytorch.GINConv(
#             apply_func=torch.nn.Linear(in_feats, out_feats),
#             aggregator_type=aggregator_type,
#             init_eps=init_eps,
#             learn_eps=learn_eps
#         )
#
#     def reset_parameters(self):
#         """Reinitialize model parameters."""
#         self.gnn.reset_parameters()
#
#     def forward(self, graph: dgl.DGLGraph, inputs: torch.Tensor) -> torch.Tensor:
#         """Update node representations.
#
#         Args:
#             graph: The batch of graphs to operate on.
#             inputs: The inputs to the layers with shape=(n_nodes, in_feats).
#
#         Returns
#             The output hidden features with shape=(n_nodes, out_feats).
#         """
#         return self.activation(self.gcn(graph, self.dropout(inputs)))
#

# class GINConvStack(GCNStack[GINConv, GINConvAggregatorType]):
#     """A wrapper around a stack of GINConv graph convolutional layers
#     """
#
#     @classmethod
#     def _default_aggregator_type(cls) -> GINConvAggregatorType:
#         return "sum"
#
#     @classmethod
#     def _gcn_factory(
#         cls,
#         in_feats: int,
#         out_feats: int,
#         aggregator_type: GINConvAggregatorType,
#         dropout: float,
#         activation: ActivationFunction,
#         init_eps: float = 0.0,
#         learn_eps: bool = False,
#     ) -> GINConv:
#
#         return GINConv(
#             in_feats=in_feats,
#             out_feats=out_feats,
#             activation=activation,
#             dropout=dropout,
#             aggregator_type=aggregator_type,
#             init_eps=init_eps,
#             learn_eps=learn_eps
#         )


class SAGEConvStack(GCNStack[dgl.nn.pytorch.SAGEConv, SAGEConvAggregatorType]):
    """A wrapper around a stack of SAGEConv graph convolutional layers"""

    @classmethod
    def _default_aggregator_type(cls) -> SAGEConvAggregatorType:
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
