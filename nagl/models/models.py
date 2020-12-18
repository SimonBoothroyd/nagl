from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import dgl.function
import torch.nn.functional
from pydantic import BaseModel, Field
from typing_extensions import Literal

from nagl.nn import SequentialLayers
from nagl.nn.gcn import SAGEConvStack
from nagl.nn.pooling import PoolingLayer
from nagl.nn.process import PostprocessLayer

ActivationFunction = Callable[[torch.tensor], torch.Tensor]

_GRAPH_ARCHITECTURES = {
    "SAGEConv": SAGEConvStack,
    # "GINConv": GINConvStack,
}


class ConvolutionConfig(BaseModel):
    """Configuration options for the convolution layers in a GCN model."""

    architecture: Literal["SAGEConv"] = Field(
        "GINConv", description="The graph convolutional architecture to use."
    )

    in_feats: int = Field(..., description="The number of input node features.")

    hidden_feats: List[int] = Field(
        ...,
        description="``hidden_feats[i]`` gives the size of node representations after "
        "the i-th GCN layer. ``len(hidden_feats)`` equals the number of GCN layers.",
    )
    activation: Optional[List[ActivationFunction]] = Field(
        None,
        description="If not None, ``activation[i]`` gives the activation function to "
        "be used for the i-th GCN layer. ``len(activation)`` equals the number of GCN "
        "layers.",
    )
    dropout: Optional[List[float]] = Field(
        None,
        description="``dropout[i]`` decides the dropout probability on the output of "
        "the i-th GCN layer. ``len(dropout)`` equals the number of GCN layers.",
    )


@dataclass
class ReadoutConfig:
    """
    Args:
        pooling_layer: The pooling layer which will concatenate the node features
            computed by a graph convolution into appropriate extended features (e.g.
            bond or angle features).

        postprocess_layer: An optional layer to apply to the final readout.

        hidden_feats: ``hidden_feats[i]`` gives the size of node representations
            after the i-th layer. ``len(hidden_feats)`` equals the number of layers.

        activation: ``activation[i]`` decides the activation function to apply to the
            i-th layer. ``len(activation)`` equals the number of layers. In no values
            are specified ReLU will be used after each layer.

        dropout: ``dropout[i]`` decides the dropout probability on the output of the
            i-th layer. ``len(dropout)`` equals the number of layers. If no values
            are specified then do dropout will take place.
    """

    pooling_layer: PoolingLayer

    hidden_feats: List[int]
    activation: Optional[List[torch.nn.Module]] = None
    dropout: Optional[List[float]] = None

    postprocess_layer: Optional[PostprocessLayer] = None


class MolGraph(torch.nn.Module):
    """A model which applies a graph convolutional step followed by multiple (labelled)
    pooling and readout steps.
    """

    def __init__(
        self,
        convolution_config: ConvolutionConfig,
        readout_configs: Dict[str, ReadoutConfig],
    ):

        super(MolGraph, self).__init__()

        self.convolution = _GRAPH_ARCHITECTURES[convolution_config.architecture](
            in_feats=convolution_config.in_feats,
            hidden_feats=convolution_config.hidden_feats,
            activation=convolution_config.activation,
            dropout=convolution_config.dropout,
        )

        self._pooling_layers: Dict[str, PoolingLayer] = {
            readout_type: readout_config.pooling_layer
            for readout_type, readout_config in readout_configs.items()
        }
        self._postprocess_layers: Dict[str, PostprocessLayer] = {
            readout_type: readout_config.postprocess_layer
            for readout_type, readout_config in readout_configs.items()
            if readout_config.postprocess_layer is not None
        }

        self._readouts: Dict[str, SequentialLayers] = {
            readout_type: SequentialLayers(
                (
                    convolution_config.hidden_feats[-1]
                    * readout_config.pooling_layer.n_feature_columns()
                ),
                readout_config.hidden_feats,
                activation=readout_config.activation,
                dropout=readout_config.dropout,
            )
            for readout_type, readout_config in readout_configs.items()
        }

        # Add the layers directly to the model. This is required for pytorch to detect
        # the parameters of the child models.
        for readout_type, pooling_layer in self._pooling_layers.items():
            setattr(self, f"pooling_{readout_type}", pooling_layer)

        for readout_type, readout_layer in self._readouts.items():
            setattr(self, f"readout_{readout_type}", readout_layer)

    def forward(
        self, graph: dgl.DGLGraph, inputs: torch.Tensor
    ) -> Dict[str, torch.Tensor]:

        # The input graph will be heterogenous - the edges are split into forward
        # edge types and their symmetric reverse counterparts. The convultion layer
        # doesn't need this information and hence we produce a homogeneous graph for
        # it to operate on with only a single edge type.
        homo_graph = dgl.to_homogeneous(graph, ndata=["feat"], edata=["feat"])

        graph.ndata["h"] = self.convolution(homo_graph, inputs)

        # The pooling, readout and processing layers then operate on the fully edge
        # annotated graph.
        readouts: Dict[str, torch.Tensor] = {
            readout_type: readout(self._pooling_layers[readout_type].forward(graph))
            for readout_type, readout in self._readouts.items()
        }

        for layer_name, layer in self._postprocess_layers.items():
            readouts[layer_name] = layer.forward(graph, readouts[layer_name])

        return readouts
