from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import dgl.function
import torch.nn.functional
from dgllife.model import GraphSAGE

from nagl.nn import SequentialLayers
from nagl.nn.pooling import PoolingLayer
from nagl.nn.process import PostprocessLayer

ActivationFunction = Callable[[torch.tensor, bool], torch.Tensor]


@dataclass
class ConvolutionConfig:
    """
    Args:
        in_feats: Number of input node features.

        hidden_feats: ``hidden_feats[i]`` gives the size of node representations after
            the i-th GraphSAGE layer. ``len(hidden_feats)`` equals the number of
            GraphSAGE layers.

        activation: If not None, ``activation[i]`` gives the activation function to be
            used for the i-th GraphSAGE layer. ``len(activation)`` equals the number of
            GraphSAGE layers.

        dropout: ``dropout[i]`` decides the dropout probability on the output of the i-th
            GraphSAGE layer. ``len(dropout)`` equals the number of GraphSAGE layers.
    """

    in_feats: int
    hidden_feats: List[int]
    activation: Optional[List[ActivationFunction]] = None
    dropout: Optional[List[float]] = None


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


class MolSAGE(torch.nn.Module):
    """A models which applies a graph convolutional step followed by multiple (labelled)
    pooling and readout steps.
    """

    def __init__(
        self,
        convolution_config: ConvolutionConfig,
        readout_configs: Dict[str, ReadoutConfig],
    ):

        super(MolSAGE, self).__init__()

        self.convolution = GraphSAGE(
            in_feats=convolution_config.in_feats,
            hidden_feats=convolution_config.hidden_feats,
            activation=convolution_config.activation,
            aggregator_type=["mean"] * len(convolution_config.hidden_feats),
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
