import typing

import pytorch_lightning as pl
import torch
import torch.nn

import nagl.nn
import nagl.nn.modules
import nagl.nn.pooling
import nagl.nn.postprocess
from nagl.config import Config
from nagl.config.data import DatasetTarget
from nagl.config.model import ActivationFunction
from nagl.molecules import DGLMolecule, DGLMoleculeBatch
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

        self.convolution_module = nagl.nn.modules.ConvolutionModule(
            config.model.convolution.type,
            n_input_feats,
            config.model.convolution.hidden_feats,
            _get_activation(config.model.convolution.activation),
            config.model.convolution.dropout,
        )
        self.readout_modules = torch.nn.ModuleDict(
            {
                readout_name: nagl.nn.modules.ReadoutModule(
                    pooling_layer=nagl.nn.pooling.get_pooling_layer(
                        readout_config.pooling
                    )(),
                    readout_layers=nagl.nn.Sequential(
                        config.model.convolution.hidden_feats[-1],
                        readout_config.readout.hidden_feats,
                        _get_activation(readout_config.readout.activation),
                        readout_config.readout.dropout,
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

        self.convolution_module(molecule)

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
        targets: typing.Dict[str, DatasetTarget] = dataset_configs[step_type].targets

        y_pred = self.forward(molecule)
        metric = torch.zeros(1).type_as(next(iter(y_pred.values())))

        for target_name, target_config in targets.items():

            if target_name not in labels:
                continue

            target_labels = labels[target_name]
            target_y_pred = y_pred[target_config.readout]

            metric_function = get_metric(target_config.metric)

            target_metric = metric_function(target_y_pred, target_labels)
            self.log(f"{step_type}_{target_name}_{target_config.metric}", target_metric)

            metric += target_metric

        self.log(f"{step_type}_loss", metric)
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


# class DGLMoleculeDataModule(pl.LightningDataModule):
#     """A utility class that makes loading and featurizing train, validation and test
#     sets more compact."""
#
#     def __init__(
#         self,
#         config: Config,
#         cache_dir: typing.Optional[pathlib.Path] = None,
#         molecule_to_dgl: typing.Optional[MoleculeToDGLFunc] = None,
#     ):
#         """
#
#         Args:
#             config: The configuration defining what data should be included.
#             cache_dir: The (optional) directory to store and load cached featurized data
#                 in. **No validation is done to ensure the loaded data matches the input
#                 config so be extra careful when using this option**.
#             molecule_to_dgl: A (optional) callable to use when converting an OpenFF
#                 ``Molecule`` object to a ``DGLMolecule`` object. By default, the
#                 ``DGLMolecule.from_openff`` class method is used.
#         """
#         super().__init__()
#
#         self._config = config
#         self._cache_dir = cache_dir
#
#         self._molecule_to_dgl = molecule_to_dgl
#
#         self._data_sets: typing.Dict[str, ConcatDataset] = {}
#
#         self._data_set_paths = {
#             "train": self._create_dataloader(self._config.data.training, "train"),
#             "val": self._create_dataloader(self._config.data.validation, "val"),
#             "test": self._create_dataloader(self._config.data.test, "test"),
#         }
#
#     def _create_dataloader(
#         self,
#         dataset_config: typing.Optional[Dataset],
#         stage: typing.Literal["train", "val", "test"],
#     ) -> typing.Optional[typing.List[pathlib.Path]]:
#
#         if dataset_config is None:
#             return None
#
#         if len(dataset_config.targets) != 1:
#
#             raise NotImplementedError(
#                 "Exactly one target per stage (train, val, test) is currently supported."
#             )
#
#         target_name, target = next(iter(dataset_config.targets.items()))
#
#         def _factory() -> DGLMoleculeDataLoader:
#
#             target_data = self._data_sets[stage]
#
#             return DGLMoleculeDataLoader(
#                 target_data,
#                 batch_size=(
#                     target.batch_size
#                     if target.batch_size is not None
#                     else len(target_data)
#                 ),
#             )
#
#         setattr(self, f"{stage}_dataloader", _factory)
#
#         return [pathlib.Path(source) for source in target.sources]
#
#     def prepare_data(self):
#         pass
#         # for stage, stage_paths in self._data_set_paths.items():
#         #
#         #     if stage_paths is None:
#         #         continue
#         #
#         #     self._data_sets[stage] = ConcatDataset(
#         #         DGLMoleculeDataset.from_file(
#         #             data_path,
#         #             atom_features=self._config.model.atom_features,
#         #             bond_features=self._config.model.bond_features,
#         #             molecule_to_dgl=self._molecule_to_dgl,
#         #         )
#         #         for data_path in stage_paths
#         #     )
#
#     def setup(self, stage: typing.Optional[str] = None):
#
#         data_set_paths = (
#             self._data_set_paths
#             if stage is None
#             else {stage: self._data_set_paths[stage]}
#         )
#
#         for stage, stage_paths in data_set_paths.items():
#
#             if stage_paths is None:
#                 continue
#
#             self._data_sets[stage] = ConcatDataset(
#                 DGLMoleculeDataset.from_file(
#                     data_path,
#                     atom_features=self._config.model.atom_features,
#                     bond_features=self._config.model.bond_features,
#                     molecule_to_dgl=self._molecule_to_dgl,
#                 )
#                 for data_path in stage_paths
#             )
