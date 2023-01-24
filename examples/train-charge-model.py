import logging
from typing import Dict

import pytorch_lightning as pl
import torch
from openff.toolkit.topology import Molecule
from openff.units import unit
from torch.utils.data import DataLoader

from nagl.config import Config, DataConfig, ModelConfig, OptimizerConfig
from nagl.config.data import Dataset, Target
from nagl.config.model import GCNConvolutionModule, ReadoutModule, Sequential
from nagl.datasets import DGLMoleculeDataset, collate_dgl_molecules
from nagl.features import AtomConnectivity, AtomicElement
from nagl.training import DGLMoleculeLightningModel


def label_function(molecule: Molecule) -> Dict[str, torch.Tensor]:
    """Generates a set of train / val / test labels for a given molecule."""

    logging.info(f"labelling {molecule.to_smiles(explicit_hydrogens=False)}")

    molecule.assign_partial_charges("am1-mulliken")
    partial_charges = molecule.partial_charges.m_as(unit.elementary_charge)

    return {"charges-am1": torch.from_numpy(partial_charges).float()}


def main():

    logging.basicConfig(level=logging.INFO)

    print(torch.seed())

    # Define the model.
    n_gcn_layers = 4
    n_gcn_hidden_features = 128

    n_am1_layers = 3
    n_am1_hidden_features = 64

    learning_rate = 0.001

    training_config = Config(
        model=ModelConfig(
            # Define the atom / bond features of interest.
            atom_features=[AtomicElement(values=["C", "O", "H"]), AtomConnectivity()],
            bond_features=[],
            convolution=GCNConvolutionModule(
                type="SAGEConv",
                hidden_feats=[n_gcn_hidden_features] * n_gcn_layers,
                activation=["ReLU"] * n_gcn_layers,
            ),
            readouts={
                # The keys of the readout nn should correspond to keys in the
                # label dictionary.
                "charges": ReadoutModule(
                    pooling="atom",
                    forward=Sequential(
                        hidden_feats=[n_am1_hidden_features] * n_am1_layers + [2],
                        activation=["ReLU"] * n_am1_layers + ["Identity"],
                    ),
                    postprocess="charges",
                )
            },
        ),
        data=DataConfig(
            training=Dataset(
                # column must correspond to the name of one of the labels and
                # readout one of the model readouts.
                targets=[Target(column="charges-am1", readout="charges", metric="rmse")]
            ),
            test=Dataset(
                targets=[Target(column="charges-am1", readout="charges", metric="rmse")]
            ),
        ),
        optimizer=OptimizerConfig(type="Adam", lr=learning_rate),
    )

    model = DGLMoleculeLightningModel(training_config)
    print(model)

    # Load in the training and test data
    training_data = DGLMoleculeDataset.from_smiles(
        ["CO", "CCO", "CCCO", "CCCCO"],
        training_config.model.atom_features,
        training_config.model.bond_features,
        label_function,
    )
    training_loader = DataLoader(
        training_data,
        batch_size=len(training_data),
        shuffle=False,
        collate_fn=collate_dgl_molecules,
    )

    test_data = DGLMoleculeDataset.from_smiles(
        ["CCCCCCCCCO"],
        training_config.model.atom_features,
        training_config.model.bond_features,
        label_function,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=len(test_data),
        shuffle=False,
        collate_fn=collate_dgl_molecules,
    )

    # Train the model
    n_epochs = 100

    n_gpus = 0 if not torch.cuda.is_available() else 1
    print(f"Using {n_gpus} GPUs")

    trainer = pl.Trainer(gpus=n_gpus, min_epochs=n_epochs, max_epochs=n_epochs)

    trainer.fit(model, train_dataloaders=training_loader)
    trainer.test(model, dataloaders=test_loader)


if __name__ == "__main__":
    main()
