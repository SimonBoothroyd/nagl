import pathlib
import typing

import pyarrow.parquet
import pytorch_lightning as pl
import torch
import torch.nn
from torch.utils.data import DataLoader

import nagl.nn
import nagl.nn.convolution
import nagl.nn.pooling
import nagl.nn.postprocess
import nagl.nn.readout
from nagl.config import Config
from nagl.config.data import Dataset as DatasetConfig
from nagl.config.model import ActivationFunction
from nagl.datasets import DGLMoleculeDataset, collate_dgl_molecules
from nagl.molecules import DGLMolecule, DGLMoleculeBatch, MoleculeToDGLFunc
from nagl.training.metrics import get_metric


def _get_activation(
    types: typing.Optional[typing.List[ActivationFunction]],
) -> typing.Optional[typing.List[torch.nn.Module]]:

    return (
        None
        if types is None
        else [nagl.nn.get_activation_func(type_)() for type_ in types]
    )


class DGLMoleculeLightningModel(pl.LightningModule):
    """A model which applies a graph convolutional step followed by multiple (labelled)
    pooling and readout steps.
    """

    def __init__(self, config: Config):

        super().__init__()

        self.config = config

        n_input_feats = sum(len(feature) for feature in self.config.model.atom_features)

        convolution_class = nagl.nn.convolution.get_convolution_layer(
            config.model.convolution.type
        )
        self.convolution_module = convolution_class(
            n_input_feats,
            config.model.convolution.hidden_feats,
            _get_activation(config.model.convolution.activation),
            config.model.convolution.dropout,
        )
        self.readout_modules = torch.nn.ModuleDict(
            {
                readout_name: nagl.nn.readout.ReadoutModule(
                    pooling_layer=nagl.nn.pooling.get_pooling_layer(
                        readout_config.pooling
                    )(),
                    forward_layers=nagl.nn.Sequential(
                        config.model.convolution.hidden_feats[-1],
                        readout_config.forward.hidden_feats,
                        _get_activation(readout_config.forward.activation),
                        readout_config.forward.dropout,
                    ),
                    postprocess_layer=nagl.nn.postprocess.get_postprocess_layer(
                        readout_config.postprocess
                    )(),
                )
                for readout_name, readout_config in config.model.readouts.items()
            }
        )

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

    def _default_step(
        self,
        batch: typing.Tuple[DGLMolecule, typing.Dict[str, torch.Tensor]],
        step_type: typing.Literal["train", "val", "test"],
    ):

        molecule, labels = batch

        dataset_configs = {
            "train": self.config.data.training,
            "val": self.config.data.validation,
            "test": self.config.data.test,
        }
        targets = dataset_configs[step_type].targets

        y_pred = self.forward(molecule)
        metric = torch.zeros(1).type_as(next(iter(y_pred.values())))

        for target in targets:

            if labels[target.column] is None:
                continue

            target_labels = labels[target.column]
            target_y_pred = y_pred[target.readout]

            metric_function = get_metric(target.metric)

            target_metric = metric_function(target_y_pred, target_labels)
            self.log(f"{step_type}-{target.column}-{target.metric}", target_metric)

            metric += target_metric

        self.log(f"{step_type}-loss", metric)
        return metric

    def training_step(self, train_batch, batch_idx):
        return self._default_step(train_batch, "train")

    def validation_step(self, val_batch, batch_idx):
        return self._default_step(val_batch, "val")

    def test_step(self, test_batch, batch_idx):
        return self._default_step(test_batch, "test")

    def configure_optimizers(self):

        if self.config.optimizer.type.lower() != "adam":
            raise NotImplementedError

        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.optimizer.lr)
        return optimizer


class DGLMoleculeDataModule(pl.LightningDataModule):
    """A utility class that makes loading and featurizing train, validation and test
    sets more compact."""

    def __init__(
        self,
        config: Config,
        cache_dir: typing.Optional[pathlib.Path] = None,
        molecule_to_dgl: typing.Optional[MoleculeToDGLFunc] = None,
    ):
        """

        Args:
            config: The configuration defining what data should be included.
            cache_dir: The (optional) directory to store and load cached featurized data
                in. **No validation is done to ensure the loaded data matches the input
                config so be extra careful when using this option**.
            molecule_to_dgl: A (optional) callable to use when converting an OpenFF
                ``Molecule`` object to a ``DGLMolecule`` object. By default, the
                ``DGLMolecule.from_openff`` class method is used.
        """
        super().__init__()

        self._config = config
        self._cache_dir = cache_dir

        self._molecule_to_dgl = molecule_to_dgl

        self._data_sets: typing.Dict[str, DGLMoleculeDataset] = {}

        self._data_set_configs: typing.Dict[str, typing.Optional[DatasetConfig]] = {
            "train": config.data.training,
            "val": config.data.validation,
            "test": config.data.test,
        }

        self._data_set_paths = {
            stage: None if dataset_config is None else dataset_config.sources
            for stage, dataset_config in self._data_set_configs.items()
        }

        for stage, dataset_config in self._data_set_configs.items():
            self._create_dataloader(dataset_config, stage)

    def _create_dataloader(
        self,
        dataset_config: typing.Optional[DatasetConfig],
        stage: typing.Literal["train", "val", "test"],
    ):

        if dataset_config is None:
            return

        def _factory() -> DataLoader:

            target_data = self._data_sets[stage]

            batch_size = (
                len(target_data)
                if dataset_config.batch_size is None
                else dataset_config.batch_size
            )

            return DataLoader(
                dataset=target_data,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_dgl_molecules,
            )

        setattr(self, f"{stage}_dataloader", _factory)

    def prepare_data(self):

        for stage, stage_paths in self._data_set_paths.items():

            if stage_paths is None:
                continue

            dataset_config = self._data_set_configs[stage]
            columns = sorted({target.column for target in dataset_config.targets})

            dataset = DGLMoleculeDataset.from_unfeaturized(
                [pathlib.Path(path) for path in stage_paths],
                columns=columns,
                atom_features=self._config.model.atom_features,
                bond_features=self._config.model.bond_features,
                molecule_to_dgl=self._molecule_to_dgl,
            )

            if self._cache_dir is None:
                self._data_sets[stage] = dataset
            else:
                self._cache_dir.mkdir(parents=True, exist_ok=True)

                table = dataset.to_table()
                pyarrow.parquet.write_table(table, self._cache_dir / f"{stage}.parquet")

    def setup(self, stage: typing.Optional[str] = None):

        if self._cache_dir is None:
            return

        for stage in self._data_set_paths:

            self._data_sets[stage] = DGLMoleculeDataset.from_featurized(
                self._cache_dir / f"{stage}.parquet", columns=None
            )
