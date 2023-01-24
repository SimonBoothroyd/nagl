import typing

import torch.nn.functional

from nagl.molecules import DGLMolecule, DGLMoleculeBatch
from nagl.nn.convolution import ConvolutionModule
from nagl.nn.readout import ReadoutModule


class DGLMoleculeModel(torch.nn.Module):
    """A model which applies a convolutional step followed by multiple pooling and
    readout steps.
    """

    def __init__(
        self,
        convolution_module: ConvolutionModule,
        readout_modules: typing.Dict[str, ReadoutModule],
    ):

        super(DGLMoleculeModel, self).__init__()

        self.convolution_module = convolution_module
        self.readout_modules = torch.nn.ModuleDict(readout_modules)

    def forward(
        self, molecule: typing.Union[DGLMolecule, DGLMoleculeBatch]
    ) -> typing.Dict[str, torch.Tensor]:

        molecule.graph.ndata["h"] = self.convolution_module(
            molecule.graph, molecule.atom_features
        )
        readouts: typing.Dict[str, torch.Tensor] = {
            readout_type: readout_module(molecule)
            for readout_type, readout_module in self.readout_modules.items()
        }

        return readouts
