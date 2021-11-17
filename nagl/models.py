from typing import Dict, Union

import torch.nn.functional
from pydantic import BaseModel, Field
from typing_extensions import Literal

from nagl.molecules import DGLMolecule, DGLMoleculeBatch
from nagl.nn import SequentialConfig, SequentialLayers
from nagl.nn.gcn import SAGEConvStack
from nagl.nn.pooling import PoolAtomFeatures, PoolBondFeatures, PoolingLayer
from nagl.nn.process import ComputePartialCharges, PostprocessLayer

_GRAPH_ARCHITECTURES = {
    "SAGEConv": SAGEConvStack,
    # "GINConv": GINConvStack,
}
_POOLING_LAYERS = {
    "PoolAtomFeatures": PoolAtomFeatures,
    "PoolBondFeatures": PoolBondFeatures,
}
_POSTPROCESS_LAYERS = {
    "ComputePartialCharges": ComputePartialCharges,
}


class ConvolutionConfig(SequentialConfig):
    """Configuration options for the convolution layers in a GCN model."""

    architecture: Literal["SAGEConv"] = Field(
        "SAGEConv", description="The graph convolutional architecture to use."
    )


class ReadoutConfig(BaseModel):
    """Configuration options for the readout layers of a GCN based model."""

    pooling_layer: Union[PoolAtomFeatures.Config, PoolBondFeatures.Config] = Field(
        ...,
        description="The pooling layer which will concatenate the node features "
        "computed by a graph convolution into appropriate extended features (e.g. bond "
        "or angle features). The concatenated features will be provided as input to the "
        "dense readout layers.",
    )
    readout_layers: SequentialConfig = Field(
        ...,
        description="The dense NN readout layers to apply to the output of the pooling"
        "layers.",
    )
    postprocess_layer: Union[ComputePartialCharges.Config] = Field(
        None,
        description="An optional postprocessing layer to apply to the output of "
        "the readout layers.",
    )


class MoleculeGCNModel(torch.nn.Module):
    """A model which applies a graph convolutional step followed by multiple (labelled)
    pooling and readout steps.
    """

    def __init__(
        self,
        convolution_config: ConvolutionConfig,
        readout_configs: Dict[str, ReadoutConfig],
    ):

        super(MoleculeGCNModel, self).__init__()

        self._convolution = _GRAPH_ARCHITECTURES[convolution_config.architecture](
            in_feats=convolution_config.in_feats,
            hidden_feats=convolution_config.hidden_feats,
            activation=convolution_config.activation,
            dropout=convolution_config.dropout,
        )

        self._pooling_layers: Dict[str, PoolingLayer] = {
            readout_type: _POOLING_LAYERS[
                readout_config.pooling_layer.type
            ].from_config(readout_config.pooling_layer)
            for readout_type, readout_config in readout_configs.items()
        }
        self._readouts: Dict[str, SequentialLayers] = {
            readout_type: SequentialLayers.from_config(readout_config.readout_layers)
            for readout_type, readout_config in readout_configs.items()
        }
        self._postprocess_layers: Dict[str, PostprocessLayer] = {
            readout_type: _POSTPROCESS_LAYERS[
                readout_config.postprocess_layer.type
            ].from_config(readout_config.postprocess_layer)
            for readout_type, readout_config in readout_configs.items()
            if readout_config.postprocess_layer is not None
        }

        # Add the layers directly to the model. This is required for pytorch to detect
        # the parameters of the child models.
        for readout_type, pooling_layer in self._pooling_layers.items():
            setattr(self, f"pooling_{readout_type}", pooling_layer)

        for readout_type, readout_layer in self._readouts.items():
            setattr(self, f"readout_{readout_type}", readout_layer)

        for postprocess_type, postprocess_layer in self._postprocess_layers.items():
            setattr(self, f"postprocess_{postprocess_type}", postprocess_layer)

    def forward(
        self, molecule: Union[DGLMolecule, DGLMoleculeBatch]
    ) -> Dict[str, torch.Tensor]:

        # The input graph will be heterogeneous - the edges are split into forward
        # edge types and their symmetric reverse counterparts. The convolution layer
        # doesn't need this information and hence we produce a homogeneous graph for
        # it to operate on with only a single edge type.
        molecule.graph.ndata["h"] = self._convolution(
            molecule.homograph, molecule.atom_features
        )

        # The pooling, readout and processing layers then operate on the fully edge
        # annotated graph.
        readouts: Dict[str, torch.Tensor] = {
            readout_type: readout(self._pooling_layers[readout_type].forward(molecule))
            for readout_type, readout in self._readouts.items()
        }

        for layer_name, layer in self._postprocess_layers.items():
            readouts[layer_name] = layer.forward(molecule, readouts[layer_name])

        return readouts
