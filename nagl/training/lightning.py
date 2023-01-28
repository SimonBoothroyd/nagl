import dataclasses
import hashlib
import json
import logging
import pathlib
import subprocess
import tempfile
import typing

import pyarrow.parquet
import pydantic
import pytorch_lightning as pl
import torch
import torch.nn
from pytorch_lightning.loggers import MLFlowLogger
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
from nagl.features import AtomFeature, BondFeature
from nagl.molecules import DGLMolecule, DGLMoleculeBatch, MoleculeToDGLFunc
from nagl.training.metrics import get_metric

_BatchType = typing.Tuple[
    typing.Union[DGLMolecule, DGLMoleculeBatch], typing.Dict[str, torch.Tensor]
]

_logger = logging.getLogger(__name__)


def _get_activation(
    types: typing.Optional[typing.List[ActivationFunction]],
) -> typing.Optional[typing.List[torch.nn.Module]]:

    return (
        None
        if types is None
        else [nagl.nn.get_activation_func(type_)() for type_ in types]
    )


def _hash_featurized_dataset(
    dataset_config: DatasetConfig,
    atom_features: typing.List[AtomFeature],
    bond_features: typing.List[BondFeature],
) -> str:
    """A quick and dirty way to hash a 'featurized' dataset.

    Args:
        dataset_config: The dataset configuration.
        atom_features: The atom feature set.
        bond_features: The bond feature set.

    Returns:
        The dataset hash.
    """

    @pydantic.dataclasses.dataclass
    class DatasetHash:

        atom_features: typing.List[AtomFeature]
        bond_features: typing.List[BondFeature]

        columns: typing.List[str]
        source_hashes: typing.List[str]

    source_hashes = []

    if dataset_config.sources is not None:

        for source in dataset_config.sources:

            result = subprocess.run(
                ["openssl", "sha256", source], capture_output=True, text=True
            )
            result.check_returncode()

            source_hashes.append(result.stdout)

    columns = sorted({target.column for target in dataset_config.targets})

    dataset_hash = DatasetHash(
        atom_features=atom_features,
        bond_features=bond_features,
        source_hashes=source_hashes,
        columns=columns,
    )
    dataset_hash_json = json.dumps(dataclasses.asdict(dataset_hash), sort_keys=True)

    return hashlib.sha256(dataset_hash_json.encode()).hexdigest()


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
        batch: _BatchType,
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
            self.log(f"{step_type}/{target.column}/{target.metric}", target_metric)

            metric += target_metric

        self.log(f"{step_type}/loss", metric)
        return metric

    def training_step(self, train_batch, batch_idx):
        return self._default_step(train_batch, "train")

    def validation_step(self, val_batch, batch_idx):
        return self._default_step(val_batch, "val")

    def test_step(self, test_batch, batch_idx):

        metric = self._default_step(test_batch, "test")

        if isinstance(self.logger, MLFlowLogger):
            self._log_report_artifact(test_batch)

        return metric

    def _log_report_artifact(self, batch_and_labels: _BatchType):

        # prevent circular import
        from nagl.reporting import create_atom_label_report

        batch, labels = batch_and_labels

        if isinstance(batch, DGLMoleculeBatch):
            molecules = batch.unbatch()
        else:
            molecules = [batch]

        n_atoms_per_mol = [molecule.n_atoms for molecule in molecules]

        prediction = self.forward(batch)

        targets = self.config.data.test.targets

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = pathlib.Path(tmp_dir)

            for target in targets:

                if labels[target.column] is None:
                    continue

                target_pred = torch.split(prediction[target.readout], n_atoms_per_mol)
                target_ref = torch.split(labels[target.column], n_atoms_per_mol)

                report_entries = [
                    (molecule, target_pred[i], target_ref[i])
                    for i, molecule in enumerate(molecules)
                ]
                report_path = tmp_dir / f"{target.column}.html"

                create_atom_label_report(
                    report_entries,
                    metrics=["rmse"],
                    rank_by="rmse",
                    output_path=report_path,
                )

                self.logger.experiment.log_artifact(
                    self.logger.run_id, local_path=str(report_path)
                )

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

        data_set_configs = {
            "train": config.data.training,
            "val": config.data.validation,
            "test": config.data.test,
        }

        self._data_set_configs: typing.Dict[str, DatasetConfig] = {
            k: v for k, v in data_set_configs.items() if v is not None
        }
        self._data_set_paths = {
            stage: dataset_config.sources
            for stage, dataset_config in self._data_set_configs.items()
        }

        for stage, dataset_config in self._data_set_configs.items():
            self._create_dataloader(dataset_config, stage)

    def _create_dataloader(
        self,
        dataset_config: DatasetConfig,
        stage: typing.Literal["train", "val", "test"],
    ):
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

            dataset_config = self._data_set_configs[stage]
            columns = sorted({target.column for target in dataset_config.targets})

            hash_string = (
                None
                if self._cache_dir is None
                else _hash_featurized_dataset(
                    dataset_config,
                    self._config.model.atom_features,
                    self._config.model.bond_features,
                )
            )
            cached_path: pathlib.Path = (
                None
                if self._cache_dir is None
                else self._cache_dir / f"{stage}-{hash_string}.parquet"
            )

            if self._cache_dir is not None and cached_path.is_file():
                _logger.info(f"found cached featurized dataset at {cached_path}")
                continue

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
                pyarrow.parquet.write_table(dataset.to_table(), cached_path)

    def setup(self, stage: typing.Optional[str] = None):

        if self._cache_dir is None:
            return

        for stage in self._data_set_paths:

            hash_string = _hash_featurized_dataset(
                self._data_set_configs[stage],
                self._config.model.atom_features,
                self._config.model.bond_features,
            )

            self._data_sets[stage] = DGLMoleculeDataset.from_featurized(
                self._cache_dir / f"{stage}-{hash_string}.parquet", columns=None
            )
