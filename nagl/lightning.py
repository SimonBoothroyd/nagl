from typing import Dict, Literal, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn.functional

from nagl.molecules import DGLMolecule, DGLMoleculeBatch
from nagl.nn.modules import ConvolutionModule, ReadoutModule


def _rmse_loss(y_pred: torch.Tensor, label: torch.Tensor):
    return torch.sqrt(torch.nn.functional.mse_loss(y_pred, label))


class MoleculeGCNLightningModel(pl.LightningModule):
    """A model which applies a graph convolutional step followed by multiple (labelled)
    pooling and readout steps.
    """

    def __init__(
        self,
        convolution_module: ConvolutionModule,
        readout_modules: Dict[str, ReadoutModule],
        learning_rate: float,
    ):

        super().__init__()

        self.convolution_module = convolution_module
        self.readout_modules = torch.nn.ModuleDict(readout_modules)

        self.learning_rate = learning_rate

    def forward(
        self, molecule: Union[DGLMolecule, DGLMoleculeBatch]
    ) -> Dict[str, torch.Tensor]:

        self.convolution_module(molecule)

        readouts: Dict[str, torch.Tensor] = {
            readout_type: readout_module(molecule)
            for readout_type, readout_module in self.readout_modules.items()
        }

        return readouts

    def _default_step(
        self,
        batch: Tuple[DGLMolecule, Dict[str, torch.Tensor]],
        step_type: Literal["train", "val", "test"],
    ):

        molecule, labels = batch

        y_pred = self.forward(molecule)
        loss = torch.zeros(1).type_as(next(iter(y_pred.values())))

        for label_name, label in labels.items():
            loss += _rmse_loss(y_pred[label_name], label)

        self.log(f"{step_type}_loss", loss)
        return loss

    def training_step(self, train_batch, batch_idx):
        return self._default_step(train_batch, "train")

    def validation_step(self, val_batch, batch_idx):
        return self._default_step(val_batch, "val")

    def test_step(self, test_batch, batch_idx):
        return self._default_step(test_batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
