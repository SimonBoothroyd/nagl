from typing import Dict

import numpy
import pytorch_lightning as pl
import torch
from openff.toolkit.topology import Molecule

from nagl.datasets import DGLMoleculeDataLoader, DGLMoleculeDataset
from nagl.features import AtomConnectivity, AtomFormalCharge, AtomicElement, BondOrder
from nagl.lightning import DGLMoleculeLightningModel
from nagl.nn import SequentialLayers
from nagl.nn.modules import ConvolutionModule, ReadoutModule
from nagl.nn.pooling import PoolAtomFeatures
from nagl.nn.postprocess import ComputePartialCharges


def label_function(molecule: Molecule) -> Dict[str, torch.Tensor]:
    """Generates a set of train / val / test labels for a given molecule."""
    from simtk import unit

    # Generate a set of ELF10 conformers.
    molecule.generate_conformers(n_conformers=800, rms_cutoff=0.05 * unit.angstrom)
    molecule.apply_elf_conformer_selection()

    partial_charges = []

    for conformer in molecule.conformers:

        molecule.assign_partial_charges("am1-mulliken", use_conformers=[conformer])

        partial_charges.append(
            molecule.partial_charges.value_in_unit(unit.elementary_charge)
        )

    return {
        "am1-charges": torch.from_numpy(numpy.mean(partial_charges, axis=0)).float()
    }


def main():

    print(torch.seed())

    # Define the atom / bond features of interest.
    atom_features = [
        AtomicElement(["C", "O", "H"]),
        AtomConnectivity(),
        AtomFormalCharge([-1, 0, 1]),
    ]
    bond_features = [
        BondOrder(),
    ]

    # Compute the total length of the input atomic feature vector
    n_atom_features = sum(len(feature) for feature in atom_features)

    # Load in the training and test data
    training_smiles = ["CO", "CCO", "CCCO", "CCCCO"]
    training_data = DGLMoleculeDataset.from_smiles(
        training_smiles,
        atom_features,
        bond_features,
        label_function,
        enumerate_resonance=True,
    )
    training_loader = DGLMoleculeDataLoader(
        training_data, batch_size=len(training_smiles), shuffle=False
    )

    test_smiles = [
        "CCCCCCCCCO",
    ]
    test_loader = DGLMoleculeDataLoader(
        DGLMoleculeDataset.from_smiles(
            test_smiles,
            atom_features,
            bond_features,
            label_function,
            enumerate_resonance=True,
        ),
        batch_size=len(test_smiles),
        shuffle=False,
    )

    # Define the model.
    n_gcn_layers = 5
    n_gcn_hidden_features = 128

    n_am1_layers = 2
    n_am1_hidden_features = 64

    learning_rate = 0.001

    model = DGLMoleculeLightningModel(
        convolution_module=ConvolutionModule(
            architecture="SAGEConv",
            in_feats=n_atom_features,
            hidden_feats=[n_gcn_hidden_features] * n_gcn_layers,
        ),
        readout_modules={
            # The keys of the readout modules should correspond to keys in the
            # label dictionary.
            "am1-charges": ReadoutModule(
                pooling_layer=PoolAtomFeatures(),
                readout_layers=SequentialLayers(
                    in_feats=n_gcn_hidden_features,
                    hidden_feats=[n_am1_hidden_features] * n_am1_layers + [2],
                    activation=["ReLU"] * n_am1_layers + ["Identity"],
                ),
                postprocess_layer=ComputePartialCharges(),
            )
        },
        learning_rate=learning_rate,
    )

    print(model)

    # Train the model
    n_epochs = 100

    n_gpus = 0 if not torch.cuda.is_available() else 1
    print(f"Using {n_gpus} GPUs")

    trainer = pl.Trainer(gpus=n_gpus, min_epochs=n_epochs, max_epochs=n_epochs)

    trainer.fit(model, train_dataloaders=training_loader)
    trainer.test(model, test_dataloaders=test_loader)


if __name__ == "__main__":
    main()
