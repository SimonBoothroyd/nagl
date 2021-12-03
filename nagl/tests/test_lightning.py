import os.path
import pickle

import numpy
import pytest
import torch
import torch.optim
from torch.utils.data import ConcatDataset

from nagl.datasets import DGLMoleculeDataset
from nagl.features import AtomFormalCharge, AtomicElement, BondOrder
from nagl.lightning import DGLMoleculeDataModule, DGLMoleculeLightningModel
from nagl.models import ConvolutionModule, ReadoutModule
from nagl.nn import SequentialLayers
from nagl.nn.gcn import GCNStack
from nagl.nn.pooling import PoolAtomFeatures, PoolBondFeatures
from nagl.nn.postprocess import ComputePartialCharges
from nagl.storage import (
    ConformerRecord,
    MoleculeRecord,
    MoleculeStore,
    PartialChargeSet,
    WibergBondOrderSet,
)
from nagl.tests import does_not_raise


@pytest.fixture()
def mock_atom_model() -> DGLMoleculeLightningModel:

    return DGLMoleculeLightningModel(
        convolution_module=ConvolutionModule("SAGEConv", in_feats=4, hidden_feats=[4]),
        readout_modules={
            "atom": ReadoutModule(
                pooling_layer=PoolAtomFeatures(),
                readout_layers=SequentialLayers(in_feats=4, hidden_feats=[2]),
                postprocess_layer=ComputePartialCharges(),
            ),
        },
        learning_rate=0.01,
    )


class TestDGLMoleculeLightningModel:
    def test_init(self):

        model = DGLMoleculeLightningModel(
            convolution_module=ConvolutionModule(
                "SAGEConv", in_feats=1, hidden_feats=[2, 2]
            ),
            readout_modules={
                "atom": ReadoutModule(
                    pooling_layer=PoolAtomFeatures(),
                    readout_layers=SequentialLayers(
                        in_feats=2, hidden_feats=[2], activation=["Identity"]
                    ),
                    postprocess_layer=ComputePartialCharges(),
                ),
                "bond": ReadoutModule(
                    pooling_layer=PoolBondFeatures(
                        layers=SequentialLayers(in_feats=4, hidden_feats=[4])
                    ),
                    readout_layers=SequentialLayers(in_feats=4, hidden_feats=[8]),
                ),
            },
            learning_rate=0.01,
        )

        assert model.convolution_module is not None
        assert isinstance(model.convolution_module, ConvolutionModule)

        assert isinstance(model.convolution_module.gcn_layers, GCNStack)
        assert len(model.convolution_module.gcn_layers) == 2

        assert all(x in model.readout_modules for x in ["atom", "bond"])

        assert isinstance(model.readout_modules["atom"].pooling_layer, PoolAtomFeatures)
        assert isinstance(model.readout_modules["bond"].pooling_layer, PoolBondFeatures)

        assert numpy.isclose(model.learning_rate, 0.01)

    def test_forward(self, mock_atom_model, dgl_methane):

        output = mock_atom_model.forward(dgl_methane)
        assert "atom" in output

        assert output["atom"].shape == (5, 1)

    @pytest.mark.parametrize(
        "method_name", ["training_step", "validation_step", "test_step"]
    )
    def test_step(self, mock_atom_model, method_name, dgl_methane, monkeypatch):
        def mock_forward(_):
            return {"atom": torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])}

        monkeypatch.setattr(mock_atom_model, "forward", mock_forward)

        loss = getattr(mock_atom_model, method_name)(
            (dgl_methane, {"atom": torch.tensor([[2.0, 3.0, 4.0, 5.0, 6.0]])}), 0
        )
        assert torch.isclose(loss, torch.tensor([1.0]))

    def test_configure_optimizers(self, mock_atom_model):

        optimizer = mock_atom_model.configure_optimizers()
        assert isinstance(optimizer, torch.optim.Adam)
        assert torch.isclose(torch.tensor(optimizer.defaults["lr"]), torch.tensor(0.01))


class TestDGLMoleculeDataModule:
    @pytest.fixture()
    def mock_data_module(self) -> DGLMoleculeDataModule:

        return DGLMoleculeDataModule(
            atom_features=[AtomicElement(["C", "H", "Cl"]), AtomFormalCharge([0, 1])],
            bond_features=[BondOrder()],
            partial_charge_method="am1bcc",
            bond_order_method="am1",
            enumerate_resonance=True,
            train_set_path="train.sqlite",
            train_batch_size=1,
            val_set_path="val.sqlite",
            val_batch_size=2,
            test_set_path="test.sqlite",
            test_batch_size=3,
            output_path="tmp.pkl",
            use_cached_data=True,
        )

    @pytest.fixture()
    def mock_data_store(self, tmpdir) -> str:
        store_path = os.path.join(tmpdir, "store.sqlite")

        store = MoleculeStore(store_path)
        store.store(
            MoleculeRecord(
                smiles="[Cl:1][Cl:2]",
                conformers=[
                    ConformerRecord(
                        coordinates=numpy.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
                        partial_charges=[
                            PartialChargeSet(method="am1bcc", values=[1.0, -1.0])
                        ],
                        bond_orders=[
                            WibergBondOrderSet(method="am1", values=[(0, 1, 1.0)])
                        ],
                    )
                ],
            )
        )

        return store_path

    def test_init(self, mock_data_module):

        assert isinstance(mock_data_module._atom_features[0], AtomicElement)
        assert mock_data_module.n_atom_features == 5

        assert isinstance(mock_data_module._bond_features[0], BondOrder)

        assert mock_data_module._partial_charge_method == "am1bcc"
        assert mock_data_module._bond_order_method == "am1"

        assert mock_data_module._enumerate_resonance is True

        assert mock_data_module._train_set_paths == ["train.sqlite"]
        assert mock_data_module._train_batch_size == 1

        assert mock_data_module._val_set_paths == ["val.sqlite"]
        assert mock_data_module._val_batch_size == 2

        assert mock_data_module._test_set_paths == ["test.sqlite"]
        assert mock_data_module._test_batch_size == 3

        assert mock_data_module._output_path == "tmp.pkl"
        assert mock_data_module._use_cached_data is True

    def test_prepare_data_from_path(self, mock_data_module, mock_data_store):

        dataset = mock_data_module._prepare_data_from_path([mock_data_store])
        assert isinstance(dataset, ConcatDataset)

        dataset = dataset.datasets[0]
        assert isinstance(dataset, DGLMoleculeDataset)

        assert dataset.n_features == 5
        assert len(dataset) == 1

        molecule, labels = next(iter(dataset))

        assert molecule.n_atoms == 2
        assert molecule.n_bonds == 1
        assert {*labels} == {"am1bcc-charges", "am1-wbo"}

    def test_prepare_data_from_multiple_paths(self, mock_data_module, mock_data_store):

        dataset = mock_data_module._prepare_data_from_path([mock_data_store] * 2)

        assert isinstance(dataset, ConcatDataset)
        assert len(dataset.datasets) == 2
        assert len(dataset) == 2

    def test_prepare_data_from_path_error(self, mock_data_module):

        with pytest.raises(NotImplementedError, match="Only paths to SQLite"):
            mock_data_module._prepare_data_from_path("tmp.pkl")

    def test_prepare(self, tmpdir, mock_data_store):

        data_module = DGLMoleculeDataModule(
            atom_features=[AtomicElement(["Cl", "H"])],
            bond_features=[BondOrder()],
            partial_charge_method="am1bcc",
            bond_order_method="am1",
            enumerate_resonance=False,
            train_set_path=mock_data_store,
            train_batch_size=None,
            val_set_path=mock_data_store,
            test_set_path=mock_data_store,
            output_path=os.path.join(tmpdir, "tmp.pkl"),
        )
        data_module.prepare_data()

        assert os.path.isfile(data_module._output_path)

        with open(data_module._output_path, "rb") as file:
            datasets = pickle.load(file)

        assert all(isinstance(dataset, ConcatDataset) for dataset in datasets)
        assert all(dataset.datasets[0].n_features == 2 for dataset in datasets)

    @pytest.mark.parametrize(
        "use_cached_data, expected_raises",
        [(True, does_not_raise()), (False, pytest.raises(FileExistsError))],
    )
    def test_prepare_cache(
        self, use_cached_data, expected_raises, tmpdir, mock_data_store
    ):

        data_module = DGLMoleculeDataModule(
            atom_features=[AtomicElement(["Cl", "H"])],
            bond_features=[BondOrder()],
            partial_charge_method="am1bcc",
            bond_order_method="am1",
            enumerate_resonance=False,
            train_set_path=mock_data_store,
            train_batch_size=None,
            output_path=os.path.join(tmpdir, "tmp.pkl"),
            use_cached_data=use_cached_data,
        )

        with open(data_module._output_path, "wb") as file:
            pickle.dump((None, None, None), file)

        with expected_raises:
            data_module.prepare_data()

    def test_setup(self, tmpdir, mock_data_store):

        data_module = DGLMoleculeDataModule(
            atom_features=[AtomicElement(["Cl", "H"])],
            bond_features=[BondOrder()],
            partial_charge_method="am1bcc",
            bond_order_method="am1",
            enumerate_resonance=False,
            train_set_path=mock_data_store,
            train_batch_size=None,
            output_path=os.path.join(tmpdir, "tmp.pkl"),
            use_cached_data=False,
        )
        data_module.prepare_data()
        data_module.setup()

        assert isinstance(data_module._train_data.datasets[0], DGLMoleculeDataset)
        assert data_module._val_data is None
        assert data_module._test_data is None
