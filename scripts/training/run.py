import pickle
from typing import Dict, List, Tuple

import dgl
import torch
import torch.nn
import torch.nn.functional
from matplotlib import pyplot
from openforcefield.topology import Molecule

from nagl.dataset.dataset import MoleculeGraphDataLoader, MoleculeGraphDataset
from nagl.dataset.features import (
    AtomConnectivity,
    AtomFeature,
    AtomFormalCharge,
    AtomicElement,
    AtomIsInRing,
    BondFeature,
    BondIsInRing,
)
from nagl.models.models import ConvolutionConfig, MolGraph, ReadoutConfig
from nagl.nn import SequentialConfig
from nagl.nn.pooling import PoolAtomFeatures
from nagl.nn.process import ComputePartialCharges


def label_function(molecule: Molecule) -> Dict[str, torch.Tensor]:
    """Generates a set of labels for a given molecule."""
    from simtk import unit

    return {
        "am1_charges": torch.tensor(
            [
                atom.partial_charge.value_in_unit(unit.elementary_charge)
                for atom in molecule.atoms
            ],
            dtype=torch.float,
        ),
        # "am1_wbo": torch.tensor(
        #     [bond.fractional_bond_order for bond in molecule.bonds], dtype=torch.float
        # ),
    }


def load_data_sets(
    atom_features: List[AtomFeature], bond_features: List[BondFeature]
) -> Tuple[MoleculeGraphDataLoader, MoleculeGraphDataLoader, int]:
    """Loads in the train and test molecules and generates labelled, featurized graph
    representations.
    """
    from simtk import unit

    training_set_path = "train-set-large.pkl"
    test_set_path = "test-set-large.pkl"

    with open(training_set_path, "rb") as file:
        training_molecules = pickle.load(file)

    with open(test_set_path, "rb") as file:
        test_molecules = pickle.load(file)

    # For now limit to only uncharged molecules.
    training_molecules = [
        molecule
        for molecule in training_molecules
        if all(len(atom.bonds) > 0 for atom in molecule.atoms)
        if all(
            atom.formal_charge == 0 * unit.elementary_charge for atom in molecule.atoms
        )
        and all(
            abs(atom.partial_charge) < 1.5 * unit.elementary_charge
            for atom in molecule.atoms
        )
    ]
    test_molecules = [
        molecule
        for molecule in test_molecules
        if all(
            atom.formal_charge == 0 * unit.elementary_charge for atom in molecule.atoms
        )
        and all(
            abs(atom.partial_charge) < 1.5 * unit.elementary_charge
            for atom in molecule.atoms
        )
    ]

    training_data = MoleculeGraphDataset.from_molecules(
        training_molecules, atom_features, bond_features, label_function
    )
    test_data = MoleculeGraphDataset.from_molecules(
        test_molecules, atom_features, bond_features, label_function
    )

    training_set = MoleculeGraphDataLoader(training_data, batch_size=32, shuffle=True)
    test_set = MoleculeGraphDataLoader(
        test_data, batch_size=len(test_data), shuffle=False
    )

    return training_set, test_set, training_data.n_features


def main():

    # Define the features of interest.
    atom_features = [
        AtomicElement(["C", "O", "H", "N", "S", "F", "Br", "Cl", "I", "P"]),
        AtomConnectivity(),
        AtomFormalCharge([-1, 0, 1]),
        AtomIsInRing(),
    ]
    bond_features = [
        BondIsInRing(),
    ]

    # Load in the pre-processed training and test molecules and store them in
    # featurized graphs.
    training_set, test_set, n_features = load_data_sets(atom_features, bond_features)

    # Define the model.
    n_node_features = 64

    model = MolGraph(
        convolution_config=ConvolutionConfig(
            architecture="SAGEConv",
            in_feats=n_features,
            hidden_feats=[n_node_features] * 4,
        ),
        readout_configs={
            "am1_charges": ReadoutConfig(
                pooling_layer=PoolAtomFeatures.Config(),
                readout_layers=SequentialConfig(
                    in_feats=n_node_features,
                    hidden_feats=[128, 128, 128, 2],
                    activation=["ReLU", "ReLU", "ReLU", "Identity"],
                ),
                postprocess_layer=ComputePartialCharges.Config(),
            ),
            # "am1_wbo": ReadoutConfig(
            #     pooling_layer=PoolBondFeatures.Config(
            #         layers=SequentialConfig(
            #             in_feats=n_node_features * 2,
            #             hidden_feats=[n_node_features * 2],
            #         )
            #     ),
            #     readout_layers=SequentialConfig(
            #         in_feats=n_node_features * 2,
            #         hidden_feats=[256, 256, 256, 1],
            #         activation=["ReLU", "ReLU", "ReLU", "Identity"],
            #     ),
            # ),
        },
    )

    print(model)

    # Define the optimizer and the loss function.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    criterion = torch.nn.MSELoss()

    losses = []

    for epoch in range(30):

        graph: dgl.DGLGraph

        for batch, (graph, features, labels) in enumerate(training_set):

            # Perform the models forward pass.
            y_pred = model(graph, features)

            # compute loss
            loss = torch.zeros(1)

            for label_name, label in labels.items():
                loss += torch.sqrt(criterion(y_pred[label_name], label))

            losses.append(loss.detach().numpy().item())

            # backward propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(
                f"epoch={epoch} "
                f"batch={batch} "
                f"loss={loss.item():.6f} "
                f"q_tot={y_pred['am1_charges'].sum().detach().item():.4f} "
                f"q_exp={graph.ndata['formal_charge'].sum().item():.4f}"
            )

    # Compute the test accuracy
    test_graph, test_features, test_labels = next(iter(test_set))
    model.eval()

    with torch.no_grad():

        test_pred = model(test_graph, test_features)

        test_loss = 0.0

        for label_name, label in test_labels.items():
            test_loss += torch.sqrt(criterion(test_pred[label_name], label))

        print("________________")
        print(f"test loss={test_loss}")

    # Plot the training losses.
    pyplot.plot(losses)
    pyplot.savefig("train-losses.png")
    pyplot.cla()

    # Plot the predicted vs reference values.
    for label in test_labels:

        pyplot.scatter(
            test_labels[label].flatten().numpy(),
            test_pred[label].flatten().numpy(),
            label="test",
        )
        pyplot.scatter(
            labels[label].flatten().numpy(),
            y_pred[label].flatten().detach().numpy(),
            label="train",
            alpha=0.3,
        )
        pyplot.legend()
        pyplot.gcf().set_size_inches(4, 4)
        pyplot.xlabel("OpenEye")
        pyplot.ylabel("Predicted")
        pyplot.tight_layout()
        pyplot.savefig(f"{label}.png")
        pyplot.cla()


if __name__ == "__main__":
    main()
