import abc
import typing

import torch.nn
from dgl.udf import EdgeBatch

from nagl.config.model import PoolingType
from nagl.molecules import DGLMolecule, DGLMoleculeBatch
from nagl.nn import Sequential


class PoolingLayer(torch.nn.Module, abc.ABC):
    """A convenience class for pooling together node feature vectors produced by
    a graph convolutional layer.
    """

    @classmethod
    @abc.abstractmethod
    def n_feature_columns(cls):
        """The number of concatenated feature columns."""

    @abc.abstractmethod
    def forward(
        self, molecule: typing.Union[DGLMolecule, DGLMoleculeBatch]
    ) -> torch.Tensor:
        """Returns the pooled feature vector."""


class AtomPoolingLayer(PoolingLayer):
    """A convenience class for pooling the node feature vectors produced by
    a graph convolutional layer.

    This class simply returns the features "h" from the graphs node data.
    """

    @classmethod
    def n_feature_columns(cls):
        return 1

    def forward(
        self, molecule: typing.Union[DGLMolecule, DGLMoleculeBatch]
    ) -> torch.Tensor:
        return molecule.graph.ndata["h"]


class BondPoolingLayer(PoolingLayer):
    """A convenience class for pooling the node feature vectors produced by
    a graph convolutional layer into a set of symmetric bond (edge) features.
    """

    @classmethod
    def n_feature_columns(cls):
        return 2

    def __init__(self, layers: torch.Union[Sequential, torch.nn.Module]):
        super().__init__()
        self.layers = layers

    @classmethod
    def _apply_edges(cls, edges: EdgeBatch):
        h_u = edges.src["h"]
        h_v = edges.dst["h"]

        return {"h": torch.cat([h_u, h_v], 1)}

    def forward(
        self, molecule: typing.Union[DGLMolecule, DGLMoleculeBatch]
    ) -> torch.Tensor:

        graph = molecule.graph

        graph.apply_edges(self._apply_edges)

        h_forward = graph.edata["h"][graph.edata["mask"]]
        h_reverse = graph.edata["h"][~graph.edata["mask"]]

        return self.layers(h_forward) + self.layers(h_reverse)


def get_pooling_layer(type_: PoolingType) -> typing.Type[PoolingLayer]:

    if type_.lower() == "atom":
        return AtomPoolingLayer
    if type_.lower() == "bond":
        return BondPoolingLayer

    raise NotImplementedError(f"{type_} not a supported pooling layer type")
