from collections import Callable
from typing import Dict, Tuple

import torch
from matplotlib import pyplot
from torch.optim import Optimizer

from nagl.datasets import DGLMoleculeDataLoader
from nagl.models import MoleculeGCNModel
from nagl.molecules import DGLMolecule


def run_training_loop(
    model: MoleculeGCNModel,
    optimizer: Optimizer,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    training_set: DGLMoleculeDataLoader,
    n_epochs: int,
    plot: bool = True,
) -> torch.Tensor:

    losses = torch.zeros(n_epochs)

    for epoch in range(100):

        molecule: DGLMolecule

        for batch, (molecule, labels) in enumerate(training_set):

            # Perform the models forward pass.
            y_pred = model(molecule)

            # compute loss
            loss = torch.zeros(1)

            for label_name, label in labels.items():
                loss += torch.sqrt(criterion(y_pred[label_name], label))

            losses[epoch] = loss.detach()

            # backward propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"epoch={epoch} " f"batch={batch} " f"loss={loss.item():.6f} ")

    if plot:
        pyplot.plot(losses)
        pyplot.savefig("train-losses.png")
        pyplot.cla()

    return losses


def evaluate_test_loss(
    model: MoleculeGCNModel,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    test_set: DGLMoleculeDataLoader,
    plot: bool = True,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, float]:

    test_molecule, test_labels = next(iter(test_set))
    model.eval()

    with torch.no_grad():

        test_pred = model(test_molecule)

        test_loss = 0.0

        for label_name, label in test_labels.items():
            test_loss += torch.sqrt(criterion(test_pred[label_name], label))

    if plot:

        for label in test_labels:

            pyplot.scatter(
                test_labels[label].flatten().numpy(),
                test_pred[label].flatten().numpy(),
                label="test",
            )

            pyplot.legend()
            pyplot.gcf().set_size_inches(4, 4)
            pyplot.xlabel("OpenEye")
            pyplot.ylabel("Predicted")
            pyplot.tight_layout()
            pyplot.savefig(f"{label}.png")
            pyplot.cla()

    return test_labels, test_pred, test_loss
