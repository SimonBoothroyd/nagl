import typing

import torch.nn.functional

from nagl.molecules import DGLMolecule, DGLMoleculeBatch
from nagl.nn.modules import ConvolutionModule, ReadoutModule


class MoleculeGCNModel(torch.nn.Module):
    """A model which applies a graph convolutional step followed by multiple (labelled)
    pooling and readout steps.
    """

    def __init__(
        self,
        convolution_module: ConvolutionModule,
        readout_modules: typing.Dict[str, ReadoutModule],
    ):

        super(MoleculeGCNModel, self).__init__()

        self.convolution_module = convolution_module
        self.readout_modules = torch.nn.ModuleDict(readout_modules)

    def forward(
        self, molecule: typing.Union[DGLMolecule, DGLMoleculeBatch]
    ) -> typing.Dict[str, torch.Tensor]:

        self.convolution_module(molecule)

        readouts: typing.Dict[str, torch.Tensor] = {
            readout_type: readout_module(molecule)
            for readout_type, readout_module in self.readout_modules.items()
        }

        return readouts
