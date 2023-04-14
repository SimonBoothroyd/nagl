import numpy
import pyarrow
import pyarrow.parquet
import pytest
import torch
import torch.optim
from torch.utils.data import DataLoader

import nagl.nn
import nagl.nn.convolution
import nagl.nn.pooling
import nagl.nn.postprocess
import nagl.nn.readout
from nagl.config import Config, DataConfig, ModelConfig, OptimizerConfig
from nagl.config.data import Dataset, DipoleTarget, ReadoutTarget
from nagl.config.model import GCNConvolutionModule, ReadoutModule, Sequential
from nagl.datasets import DGLMoleculeDataset
from nagl.features import AtomConnectivity, AtomicElement, AtomIsInRing, BondOrder
from nagl.molecules import DGLMolecule
from nagl.training.lightning import (
    DGLMoleculeDataModule,
    DGLMoleculeLightningModel,
    _hash_featurized_dataset,
)


@pytest.fixture()
def mock_config() -> Config:
    return Config(
        model=ModelConfig(
            atom_features=[AtomConnectivity()],
            bond_features=[],
            convolution=GCNConvolutionModule(
                type="SAGEConv", hidden_feats=[4, 4], activation=["ReLU", "ReLU"]
            ),
            readouts={
                "atom": ReadoutModule(
                    pooling="atom",
                    forward=Sequential(hidden_feats=[2], activation=["Identity"]),
                    postprocess="charges",
                )
            },
        ),
        data=DataConfig(
            training=Dataset(
                sources=[""],
                targets=[
                    ReadoutTarget(column="charges-am1", readout="atom", metric="rmse")
                ],
                batch_size=4,
            ),
            validation=Dataset(
                sources=[""],
                targets=[
                    ReadoutTarget(column="charges-am1", readout="atom", metric="rmse")
                ],
                batch_size=5,
            ),
            test=Dataset(
                sources=[""],
                targets=[
                    ReadoutTarget(column="charges-am1", readout="atom", metric="rmse")
                ],
                batch_size=6,
            ),
        ),
        optimizer=OptimizerConfig(type="Adam", lr=0.01),
    )


@pytest.fixture()
def mock_config_dipole() -> Config:
    return Config(
        model=ModelConfig(
            atom_features=[AtomConnectivity()],
            bond_features=[],
            convolution=GCNConvolutionModule(
                type="SAGEConv", hidden_feats=[4, 4], activation=["ReLU", "ReLU"]
            ),
            readouts={
                "atom": ReadoutModule(
                    pooling="atom",
                    forward=Sequential(hidden_feats=[2], activation=["Identity"]),
                    postprocess="charges",
                )
            },
        ),
        data=DataConfig(
            training=Dataset(
                sources=[""],
                targets=[
                    DipoleTarget(
                        column="dipole",
                        charge_label="charges-am1",
                        conformation_label="conformation",
                        metric="rmse",
                    )
                ],
                batch_size=4,
            ),
            validation=Dataset(
                sources=[""],
                targets=[
                    DipoleTarget(
                        column="dipole",
                        charge_label="charges-am1",
                        conformation_label="conformation",
                        metric="rmse",
                    )
                ],
                batch_size=5,
            ),
            test=Dataset(
                sources=[""],
                targets=[
                    DipoleTarget(
                        column="dipole",
                        charge_label="charges-am1",
                        conformation_label="conformation",
                        metric="rmse",
                    )
                ],
                batch_size=6,
            ),
        ),
        optimizer=OptimizerConfig(type="Adam", lr=0.01),
    )


@pytest.fixture()
def mock_lightning_model(mock_config) -> DGLMoleculeLightningModel:
    return DGLMoleculeLightningModel(mock_config)


def test_hash_featurized_dataset(tmp_cwd):
    labels = pyarrow.table([["C"]], ["smiles"])
    source = str(tmp_cwd / "train.parquet")

    pyarrow.parquet.write_table(labels, source)

    config = Dataset(
        sources=[source],
        targets=[ReadoutTarget(column="label-col", readout="", metric="rmse")],
    )

    atom_features = [AtomicElement()]
    bond_features = [BondOrder()]

    hash_value_1 = _hash_featurized_dataset(config, atom_features, bond_features)
    hash_value_2 = _hash_featurized_dataset(config, atom_features, bond_features)

    assert hash_value_1 == hash_value_2

    atom_features = []

    hash_value_2 = _hash_featurized_dataset(config, atom_features, bond_features)
    assert hash_value_1 != hash_value_2
    hash_value_1 = _hash_featurized_dataset(config, atom_features, bond_features)
    assert hash_value_1 == hash_value_2

    bond_features = []

    hash_value_2 = _hash_featurized_dataset(config, atom_features, bond_features)
    assert hash_value_1 != hash_value_2
    hash_value_1 = _hash_featurized_dataset(config, atom_features, bond_features)
    assert hash_value_1 == hash_value_2

    config.targets[0].column = "label-col-2"

    hash_value_2 = _hash_featurized_dataset(config, atom_features, bond_features)
    assert hash_value_1 != hash_value_2
    hash_value_1 = _hash_featurized_dataset(config, atom_features, bond_features)
    assert hash_value_1 == hash_value_2


class TestDGLMoleculeLightningModel:
    def test_init(self, mock_config):
        model = DGLMoleculeLightningModel(mock_config)

        assert isinstance(model.convolution_module, nagl.nn.convolution.SAGEConvStack)
        assert len(model.convolution_module) == 2

        assert all(x in model.readout_modules for x in ["atom"])

        assert isinstance(
            model.readout_modules["atom"].pooling_layer,
            nagl.nn.pooling.AtomPoolingLayer,
        )
        assert isinstance(
            model.readout_modules["atom"].postprocess_layer,
            nagl.nn.postprocess.PartialChargeLayer,
        )

    def test_forward(self, mock_lightning_model, rdkit_methane):
        dgl_molecule = DGLMolecule.from_rdkit(
            rdkit_methane,
            mock_lightning_model.config.model.atom_features,
            mock_lightning_model.config.model.bond_features,
        )

        output = mock_lightning_model.forward(dgl_molecule)
        assert "atom" in output

        assert output["atom"].shape == (5, 1)

    @pytest.mark.parametrize(
        "method_name", ["training_step", "validation_step", "test_step"]
    )
    def test_step_readout(
        self, mock_lightning_model, method_name, dgl_methane, monkeypatch
    ):
        def mock_forward(_):
            return {"atom": torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])}

        monkeypatch.setattr(mock_lightning_model, "forward", mock_forward)

        loss = getattr(mock_lightning_model, method_name)(
            (
                dgl_methane,
                {"charges-am1": torch.tensor([[2.0, 3.0, 4.0, 5.0, 6.0]])},
            ),
            0,
        )
        assert torch.isclose(loss, torch.tensor([1.0]))

    def test_step_dipole(self, mock_config_dipole, rdkit_methane, monkeypatch):
        """Make sure the dipole error is correctly calculated"""
        from openff.units import unit

        mock_model = DGLMoleculeLightningModel(mock_config_dipole)

        def mock_forward(_):
            return {"charges-am1": torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])}

        monkeypatch.setattr(mock_model, "forward", mock_forward)
        dgl_methane = DGLMolecule.from_rdkit(
            molecule=rdkit_methane, atom_features=[AtomicElement()]
        )
        # coordinates in angstrom
        conformer = rdkit_methane.GetConformer().GetPositions() * unit.angstrom
        loss = mock_model.training_step(
            (
                dgl_methane,
                {
                    "charges-am1": torch.tensor([[2.0, 3.0, 4.0, 5.0, 6.0]]),
                    "dipole": torch.Tensor([[0.0, 0.0, 0.0]]),
                    "conformation": torch.Tensor([conformer.m_as(unit.bohr)]),
                },
            ),
            0,
        )
        # calculate the loss and compare with numpy
        numpy_dipole = numpy.dot(
            numpy.array([1.0, 2.0, 3.0, 4.0, 5.0]), conformer.m_as(unit.bohr)
        )
        ref_loss = numpy.sqrt(numpy.mean((numpy_dipole - numpy.array([0, 0, 0])) ** 2))
        assert numpy.isclose(loss.numpy(), ref_loss)

    def test_configure_optimizers(self, mock_lightning_model):
        optimizer = mock_lightning_model.configure_optimizers()
        assert isinstance(optimizer, torch.optim.Adam)
        assert torch.isclose(torch.tensor(optimizer.defaults["lr"]), torch.tensor(0.01))


class TestDGLMoleculeDataModule:
    def test_init(self, tmp_cwd, mock_config):
        data_module = DGLMoleculeDataModule(mock_config, cache_dir=tmp_cwd / "cache")

        for stage in ["train", "val", "test"]:
            assert stage in data_module._data_set_configs
            assert stage in data_module._data_set_paths

            assert callable(getattr(data_module, f"{stage}_dataloader"))

        data_module._data_sets["train"] = DGLMoleculeDataset([])

        loader = data_module.train_dataloader()
        assert isinstance(loader, DataLoader)
        assert loader.batch_size == 4

    def test_prepare(self, tmp_cwd, mock_config, mocker):
        mocker.patch(
            "nagl.training.lightning._hash_featurized_dataset",
            autospec=True,
            return_value="hash-val",
        )

        parquet_path = tmp_cwd / "unfeaturized.parquet"
        pyarrow.parquet.write_table(
            pyarrow.table(
                [
                    ["[O-:1][H:2]", "[H:1][H:2]"],
                    [numpy.arange(2).astype(float), numpy.zeros(2).astype(float)],
                    [numpy.arange(2).astype(float) + 2, None],
                ],
                ["smiles", "charges-am1", "charges-am1bcc"],
            ),
            parquet_path,
        )

        mock_config.data.validation = None
        mock_config.data.test = None
        mock_config.data.training.sources = [str(parquet_path)]

        data_module = DGLMoleculeDataModule(mock_config, cache_dir=tmp_cwd / "cache")
        data_module.prepare_data()

        expected_path = tmp_cwd / "cache" / "train-hash-val.parquet"
        assert expected_path.is_file()

        table = pyarrow.parquet.read_table(expected_path)
        assert len(table) == 2

        del data_module._data_sets["train"]
        assert "train" not in data_module._data_sets
        data_module.setup("train")
        assert isinstance(data_module._data_sets["train"], DGLMoleculeDataset)
        assert len(data_module._data_sets["train"]) == 2
