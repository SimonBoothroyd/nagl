import numpy
import pyarrow
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
from nagl.config.data import Dataset, Target
from nagl.config.model import GCNConvolutionModule, ReadoutModule, Sequential
from nagl.datasets import DGLMoleculeDataset
from nagl.features import AtomConnectivity, AtomicElement, AtomIsInRing
from nagl.molecules import DGLMolecule
from nagl.training.lightning import DGLMoleculeDataModule, DGLMoleculeLightningModel


@pytest.fixture()
def mock_config() -> Config:

    return Config(
        model=ModelConfig(
            atom_features=[AtomicElement(), AtomConnectivity(), AtomIsInRing()],
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
                targets=[Target(column="charges-am1", readout="atom", metric="rmse")],
                batch_size=4,
            ),
            validation=Dataset(
                sources=[""],
                targets=[Target(column="charges-am1", readout="atom", metric="rmse")],
                batch_size=5,
            ),
            test=Dataset(
                sources=[""],
                targets=[Target(column="charges-am1", readout="atom", metric="rmse")],
                batch_size=6,
            ),
        ),
        optimizer=OptimizerConfig(type="Adam", lr=0.01),
    )


@pytest.fixture()
def mock_lightning_model(mock_config) -> DGLMoleculeLightningModel:
    return DGLMoleculeLightningModel(mock_config)


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

    def test_forward(self, mock_lightning_model, openff_methane):

        dgl_molecule = DGLMolecule.from_openff(
            openff_methane,
            mock_lightning_model.config.model.atom_features,
            mock_lightning_model.config.model.bond_features,
        )

        output = mock_lightning_model.forward(dgl_molecule)
        assert "atom" in output

        assert output["atom"].shape == (5, 1)

    @pytest.mark.parametrize(
        "method_name", ["training_step", "validation_step", "test_step"]
    )
    def test_step(self, mock_lightning_model, method_name, dgl_methane, monkeypatch):
        def mock_forward(_):
            return {"atom": torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])}

        monkeypatch.setattr(mock_lightning_model, "forward", mock_forward)

        loss = getattr(mock_lightning_model, method_name)(
            (dgl_methane, {"charges-am1": torch.tensor([[2.0, 3.0, 4.0, 5.0, 6.0]])}), 0
        )
        assert torch.isclose(loss, torch.tensor([1.0]))

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

    def test_prepare(self, tmp_cwd, mock_config):

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

        expected_path = tmp_cwd / "cache" / "train.parquet"
        assert expected_path.is_file()

        table = pyarrow.parquet.read_table(expected_path)
        assert len(table) == 2

        assert "train" not in data_module._data_sets
        data_module.setup("train")
        assert isinstance(data_module._data_sets["train"], DGLMoleculeDataset)
        assert len(data_module._data_sets["train"]) == 2
