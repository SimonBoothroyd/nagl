import pytest
import torch
import torch.optim

import nagl.nn
import nagl.nn.gcn
import nagl.nn.modules
import nagl.nn.pooling
import nagl.nn.postprocess
from nagl.config import Config, DataConfig, ModelConfig, OptimizerConfig
from nagl.config.data import Dataset, DatasetTarget
from nagl.config.model import GCNConvolutionModule, ReadoutModule, Sequential
from nagl.features import AtomConnectivity, AtomicElement, AtomIsInRing
from nagl.molecules import DGLMolecule
from nagl.training.lightning import DGLMoleculeLightningModel


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
                    readout=Sequential(hidden_feats=[2], activation=["Identity"]),
                    postprocess="charges",
                )
            },
        ),
        data=DataConfig(
            training=Dataset(
                targets={"charges": DatasetTarget("atom", sources=[""], metric="rmse")},
                batch_size=1,
            ),
            validation=Dataset(
                targets={"charges": DatasetTarget("atom", sources=[""], metric="rmse")},
                batch_size=1,
            ),
            test=Dataset(
                targets={"charges": DatasetTarget("atom", sources=[""], metric="rmse")},
                batch_size=1,
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

        assert isinstance(model.convolution_module, nagl.nn.modules.ConvolutionModule)

        assert isinstance(
            model.convolution_module.gcn_layers, nagl.nn.gcn.SAGEConvStack
        )
        assert len(model.convolution_module.gcn_layers) == 2

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
            (dgl_methane, {"charges": torch.tensor([[2.0, 3.0, 4.0, 5.0, 6.0]])}), 0
        )
        assert torch.isclose(loss, torch.tensor([1.0]))

    def test_configure_optimizers(self, mock_lightning_model):

        optimizer = mock_lightning_model.configure_optimizers()
        assert isinstance(optimizer, torch.optim.Adam)
        assert torch.isclose(torch.tensor(optimizer.defaults["lr"]), torch.tensor(0.01))


# class TestDGLMoleculeDataModule:
#     @classmethod
#     def mock_molecule_to_dgl(
#         cls,
#         molecule,
#         atom_features: List[AtomFeature],
#         bond_features: List[BondFeature],
#     ) -> "DGLMolecule":
#         return DGLMolecule.from_openff(molecule, atom_features, bond_features)
#
#     @pytest.fixture()
#     def mock_data_module(self) -> DGLMoleculeDataModule:
#
#         return DGLMoleculeDataModule(
#             atom_features=[AtomicElement(["C", "H", "Cl"]), AtomFormalCharge([0, 1])],
#             bond_features=[BondOrder()],
#             partial_charge_method="am1bcc",
#             bond_order_method="am1",
#             train_set_path="train.sqlite",
#             train_batch_size=1,
#             val_set_path="val.sqlite",
#             val_batch_size=2,
#             test_set_path="test.sqlite",
#             test_batch_size=3,
#             output_path="tmp.pkl",
#             use_cached_data=True,
#             molecule_to_dgl=TestDGLMoleculeDataModule.mock_molecule_to_dgl,
#         )
#
#     @pytest.fixture()
#     def mock_data_store(self, tmpdir) -> str:
#         store_path = os.path.join(tmpdir, "store.sqlite")
#
#         store = MoleculeStore(store_path)
#         store.store(
#             MoleculeRecord(
#                 smiles="[Cl:1][Cl:2]",
#                 conformers=[
#                     ConformerRecord(
#                         coordinates=numpy.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
#                         partial_charges=[
#                             PartialChargeSet(method="am1bcc", values=[1.0, -1.0])
#                         ],
#                         bond_orders=[
#                             WibergBondOrderSet(method="am1", values=[(0, 1, 1.0)])
#                         ],
#                     )
#                 ],
#             )
#         )
#
#         return store_path
#
#     def test_init(self, mock_data_module):
#
#         assert isinstance(mock_data_module._atom_features[0], AtomicElement)
#         assert mock_data_module.n_atom_features == 5
#
#         assert isinstance(mock_data_module._bond_features[0], BondOrder)
#
#         assert mock_data_module._partial_charge_method == "am1bcc"
#         assert mock_data_module._bond_order_method == "am1"
#
#         assert (
#             mock_data_module._molecule_to_dgl
#             == TestDGLMoleculeDataModule.mock_molecule_to_dgl
#         )
#
#         assert mock_data_module._train_set_paths == ["train.sqlite"]
#         assert mock_data_module._train_batch_size == 1
#
#         assert mock_data_module._val_set_paths == ["val.sqlite"]
#         assert mock_data_module._val_batch_size == 2
#
#         assert mock_data_module._test_set_paths == ["test.sqlite"]
#         assert mock_data_module._test_batch_size == 3
#
#         assert mock_data_module._output_path == "tmp.pkl"
#         assert mock_data_module._use_cached_data is True
#
#     def test_prepare_data_from_path(self, mock_data_module, mock_data_store):
#
#         dataset = mock_data_module._prepare_data_from_path([mock_data_store])
#         assert isinstance(dataset, ConcatDataset)
#
#         dataset = dataset.datasets[0]
#         assert isinstance(dataset, DGLMoleculeDataset)
#
#         assert dataset.n_features == 5
#         assert len(dataset) == 1
#
#         molecule, labels = next(iter(dataset))
#
#         assert molecule.n_atoms == 2
#         assert molecule.n_bonds == 1
#         assert {*labels} == {"am1bcc-charges", "am1-wbo"}
#
#     def test_prepare_data_from_multiple_paths(self, mock_data_module, mock_data_store):
#
#         dataset = mock_data_module._prepare_data_from_path([mock_data_store] * 2)
#
#         assert isinstance(dataset, ConcatDataset)
#         assert len(dataset.datasets) == 2
#         assert len(dataset) == 2
#
#     def test_prepare_data_from_path_error(self, mock_data_module):
#
#         with pytest.raises(NotImplementedError, match="Only paths to SQLite"):
#             mock_data_module._prepare_data_from_path("tmp.pkl")
#
#     def test_prepare(self, tmpdir, mock_data_store):
#
#         data_module = DGLMoleculeDataModule(
#             atom_features=[AtomicElement(["Cl", "H"])],
#             bond_features=[BondOrder()],
#             partial_charge_method="am1bcc",
#             bond_order_method="am1",
#             train_set_path=mock_data_store,
#             train_batch_size=None,
#             val_set_path=mock_data_store,
#             test_set_path=mock_data_store,
#             output_path=os.path.join(tmpdir, "tmp.pkl"),
#         )
#         data_module.prepare_data()
#
#         assert os.path.isfile(data_module._output_path)
#
#         with open(data_module._output_path, "rb") as file:
#             datasets = pickle.load(file)
#
#         assert all(isinstance(dataset, ConcatDataset) for dataset in datasets)
#         assert all(dataset.datasets[0].n_features == 2 for dataset in datasets)
#
#     @pytest.mark.parametrize(
#         "use_cached_data, expected_raises",
#         [(True, does_not_raise()), (False, pytest.raises(FileExistsError))],
#     )
#     def test_prepare_cache(
#         self, use_cached_data, expected_raises, tmpdir, mock_data_store
#     ):
#
#         data_module = DGLMoleculeDataModule(
#             atom_features=[AtomicElement(["Cl", "H"])],
#             bond_features=[BondOrder()],
#             partial_charge_method="am1bcc",
#             bond_order_method="am1",
#             train_set_path=mock_data_store,
#             train_batch_size=None,
#             output_path=os.path.join(tmpdir, "tmp.pkl"),
#             use_cached_data=use_cached_data,
#         )
#
#         with open(data_module._output_path, "wb") as file:
#             pickle.dump((None, None, None), file)
#
#         with expected_raises:
#             data_module.prepare_data()
#
#     def test_setup(self, tmpdir, mock_data_store):
#
#         data_module = DGLMoleculeDataModule(
#             atom_features=[AtomicElement(["Cl", "H"])],
#             bond_features=[BondOrder()],
#             partial_charge_method="am1bcc",
#             bond_order_method="am1",
#             train_set_path=mock_data_store,
#             train_batch_size=None,
#             output_path=os.path.join(tmpdir, "tmp.pkl"),
#             use_cached_data=False,
#         )
#         data_module.prepare_data()
#         data_module.setup()
#
#         assert isinstance(data_module._train_data.datasets[0], DGLMoleculeDataset)
#         assert data_module._val_data is None
#         assert data_module._test_data is None
