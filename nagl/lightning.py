import errno
import os.path
import pickle
from typing import Dict, List, Literal, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn.functional
from torch.utils.data import ConcatDataset

from nagl.datasets import DGLMoleculeDataLoader, DGLMoleculeDataset
from nagl.features import AtomFeature, BondFeature
from nagl.molecules import DGLMolecule, DGLMoleculeBatch
from nagl.nn.modules import ConvolutionModule, ReadoutModule
from nagl.storage import ChargeMethod, MoleculeStore, WBOMethod


def _rmse_loss(y_pred: torch.Tensor, label: torch.Tensor):
    return torch.sqrt(torch.nn.functional.mse_loss(y_pred, label))


class DGLMoleculeLightningModel(pl.LightningModule):
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


class DGLMoleculeDataModule(pl.LightningDataModule):
    """A utility class that makes loading and featurizing train, validation and test
    sets more compact."""

    @property
    def n_atom_features(self) -> Optional[int]:
        return sum(len(feature) for feature in self._atom_features)

    def __init__(
        self,
        atom_features: List[AtomFeature],
        bond_features: List[BondFeature],
        partial_charge_method: Optional[ChargeMethod],
        bond_order_method: Optional[WBOMethod],
        enumerate_resonance: bool,
        train_set_path: Union[str, List[str]],
        train_batch_size: Optional[int],
        val_set_path: Optional[Union[str, List[str]]] = None,
        val_batch_size: Optional[int] = None,
        test_set_path: Optional[Union[str, List[str]]] = None,
        test_batch_size: Optional[int] = None,
        output_path: str = "nagl-data-module.pkl",
        use_cached_data: bool = False,
    ):
        """

        Args:
            atom_features: The set of atom features to compute for each molecule
            bond_features: The set of bond features to compute for each molecule
            partial_charge_method: The (optional) type of partial charges to include
                in the training labels.
            bond_order_method: The (optional) type of bond orders to include
                in the training labels.
            enumerate_resonance: Whether to enumerate the lowest energy resonance
                structures of each molecule and store each within the DGL graph
                representation.
            train_set_path: The path(s) to the training data stored in a
                SQLite molecule store. If none is specified no training will
                be performed.
            train_batch_size: The training batch size. If none is specified, all the
                data will be included in a single batch.
            val_set_path: The (optional) path(s) to the validation data stored in a
                SQLite molecule store. If none is specified no validation will
                be performed.
            val_batch_size: The validation batch size. If none is specified, all the
                data will be included in a single batch.
            test_set_path: The (optional) path(s) to the test data stored in a
                SQLite molecule store. If none is specified no testing will
                be performed.
            test_batch_size: The test batch size. If none is specified, all the
                data will be included in a single batch.
            output_path: The path to store the PICKLE file to store the prepared data to.
            use_cached_data: Whether to simply load any data module found at
                the ``output_path`` rather re-generating it using the other provided
                arguments. **No validation is done to ensure the loaded data matches
                the input arguments so be extra careful when using this option**.
                If this is false and a file is found at ``output_path`` an exception
                will be raised.
        """
        super().__init__()

        self._atom_features = atom_features
        self._bond_features = bond_features

        self._partial_charge_method = partial_charge_method
        self._bond_order_method = bond_order_method

        self._enumerate_resonance = enumerate_resonance

        self._train_set_paths = (
            [train_set_path] if isinstance(train_set_path, str) else train_set_path
        )
        self._train_batch_size = train_batch_size
        self._train_data: Optional[ConcatDataset] = None

        if self._train_set_paths is not None:

            self.train_dataloader = lambda: DGLMoleculeDataLoader(
                self._train_data,
                batch_size=(
                    self._train_batch_size
                    if self._train_batch_size is not None
                    else len(self._train_data)
                ),
            )

        self._val_set_paths = (
            [val_set_path] if isinstance(val_set_path, str) else val_set_path
        )
        self._val_batch_size = val_batch_size
        self._val_data: Optional[ConcatDataset] = None

        if self._val_set_paths is not None:

            self.val_dataloader = lambda: DGLMoleculeDataLoader(
                self._val_data,
                batch_size=(
                    self._val_batch_size
                    if self._val_batch_size is not None
                    else len(self._val_data)
                ),
            )

        self._test_set_paths = (
            [test_set_path] if isinstance(test_set_path, str) else test_set_path
        )
        self._test_batch_size = test_batch_size
        self._test_data: Optional[ConcatDataset] = None

        if self._test_set_paths is not None:

            self.test_dataloader = lambda: DGLMoleculeDataLoader(
                self._test_data,
                batch_size=(
                    self._test_batch_size
                    if self._test_batch_size is not None
                    else len(self._test_data)
                ),
            )

        self._output_path = output_path
        self._use_cached_data = use_cached_data

    def _prepare_data_from_path(self, data_paths: List[str]) -> ConcatDataset:

        datasets = []

        for data_path in data_paths:

            extension = os.path.splitext(data_path)[-1].lower()

            if extension == ".sqlite":

                dataset = DGLMoleculeDataset.from_molecule_stores(
                    MoleculeStore(data_path),
                    partial_charge_method=self._partial_charge_method,
                    bond_order_method=self._bond_order_method,
                    atom_features=self._atom_features,
                    bond_features=self._bond_features,
                    enumerate_resonance=self._enumerate_resonance,
                )

            else:

                raise NotImplementedError(
                    f"Only paths to SQLite ``MoleculeStore`` databases are supported, and not "
                    f"'{extension}' files."
                )

            datasets.append(dataset)

        return ConcatDataset(datasets)

    def prepare_data(self):

        if os.path.isfile(self._output_path):

            if not self._use_cached_data:

                raise FileExistsError(
                    errno.EEXIST, os.strerror(errno.EEXIST), self._output_path
                )

            return

        train_data, val_data, test_data = None, None, None

        if self._train_set_paths is not None:
            train_data = self._prepare_data_from_path(self._train_set_paths)

        if self._val_set_paths is not None:
            val_data = self._prepare_data_from_path(self._val_set_paths)

        if self._test_set_paths is not None:
            test_data = self._prepare_data_from_path(self._test_set_paths)

        with open(self._output_path, "wb") as file:
            pickle.dump((train_data, val_data, test_data), file)

    def setup(self, stage: Optional[str] = None):

        with open(self._output_path, "rb") as file:
            self._train_data, self._val_data, self._test_data = pickle.load(file)
