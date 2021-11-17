import abc
from typing import Union

import torch.nn
from dgl.udf import EdgeBatch
from pydantic import BaseModel, Field
from typing_extensions import Literal

from nagl.molecules import DGLMolecule, DGLMoleculeBatch
from nagl.nn import SequentialConfig, SequentialLayers


class PoolingLayer(torch.nn.Module, abc.ABC):
    """A convenience class for pooling together node feature vectors produced by
    a graph convolutional layer.
    """

    @classmethod
    @abc.abstractmethod
    def from_config(cls, config):
        """Create an instance of a pooling layer from its configuration."""

    @classmethod
    @abc.abstractmethod
    def n_feature_columns(cls):
        """The number of concatenated feature columns."""

    @abc.abstractmethod
    def forward(self, molecule: Union[DGLMolecule, DGLMoleculeBatch]) -> torch.Tensor:
        """Returns the pooled feature vector."""


class PoolAtomFeatures(PoolingLayer):
    """A convenience class for pooling the node feature vectors produced by
    a graph convolutional layer.

    This class simply returns the features "h" from the graphs node data.
    """

    class Config(BaseModel):
        """Configuration options for a ``PoolAtomFeatures`` layer."""

        type: Literal["PoolAtomFeatures"] = "PoolAtomFeatures"

    @classmethod
    def n_feature_columns(cls):
        return 1

    @classmethod
    def from_config(cls, config: "PoolAtomFeatures.Config"):
        return cls()

    def forward(self, molecule: Union[DGLMolecule, DGLMoleculeBatch]) -> torch.Tensor:
        return molecule.graph.ndata["h"]


class PoolBondFeatures(PoolingLayer):
    """A convenience class for pooling the node feature vectors produced by
    a graph convolutional layer into a set of symmetric bond (edge) features.
    """

    class Config(BaseModel):
        """Configuration options for a ``PoolBondFeatures`` layer."""

        type: Literal["PoolBondFeatures"] = "PoolBondFeatures"

        layers: SequentialConfig = Field(
            ..., description="The NN layers to apply to the bond features."
        )

    @classmethod
    def n_feature_columns(cls):
        return 2

    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    @classmethod
    def from_config(cls, config: "PoolBondFeatures.Config"):
        return cls(layers=SequentialLayers.from_config(config.layers))

    @classmethod
    def _apply_edges(cls, edges: EdgeBatch):
        h_u = edges.src["h"]
        h_v = edges.dst["h"]

        return {"h": torch.cat([h_u, h_v], 1)}

    def forward(self, molecule: Union[DGLMolecule, DGLMoleculeBatch]) -> torch.Tensor:

        graph = molecule.graph

        with graph.local_scope():

            graph.apply_edges(self._apply_edges, etype="forward")
            h_forward = graph.edges["forward"].data["h"]

        with graph.local_scope():

            graph.apply_edges(self._apply_edges, etype="reverse")
            h_reverse = graph.edges["reverse"].data["h"]

        return self.layers(h_forward) + self.layers(h_reverse)
