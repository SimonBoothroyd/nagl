"""Classes to calculate per-readout or global loss functions"""
import abc
import typing

import torch

from nagl.training.metrics import MetricType, get_metric


class _BaseTarget(abc.ABC):
    """A general target class used to evaluate the loss of a model"""

    def __init__(self, metric: MetricType, denominator: float, weight: float):
        self.metric = metric
        self.denominator = denominator
        self.weight = weight

    @abc.abstractmethod
    def evaluate_loss(
        self,
        labels: typing.Dict[str, torch.Tensor],
        prediction: typing.Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        ...

    @abc.abstractmethod
    def target_column(self) -> str:
        ...


class ReadoutTarget(_BaseTarget):
    """A basic loss function acting on a single readout property"""

    def __init__(
        self,
        metric: MetricType,
        denominator: float,
        weight: float,
        column: str,
        readout: str,
    ):
        super().__init__(metric, denominator, weight)
        self.readout = readout
        self.column = column

    def evaluate_loss(
        self,
        labels: typing.Dict[str, torch.Tensor],
        prediction: typing.Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Evaluate the metric function between the target and predicted values."""

        metric_func = get_metric(self.metric)
        target_labels = labels[self.column]
        target_y_pred = prediction[self.readout]
        return (
            metric_func(target_y_pred, target_labels) * self.weight / self.denominator
        )

    def target_column(self) -> str:
        return self.column


class DipoleTarget(_BaseTarget):
    """Calculate the dipole loss based on some predicted charges.

    Current we calculate the RMSE between the dipole vectors but other targets may work better or some combination maybe possible

    - magnitude of the distance vector between the predicted and calculated values
    - the angle between the two vectors
    - the absolute difference in magnitude of the vectors
    """

    def __init__(
        self,
        metric: MetricType,
        denominator: float,
        weight: float,
        dipole_column: str,
        conformation_column: str,
        charge_label: str,
    ):
        super().__init__(metric, denominator, weight)
        self.dipole_column = dipole_column
        self.conformation_column = conformation_column
        self.charge_label = charge_label

    def evaluate_loss(
        self,
        labels: typing.Dict[str, torch.Tensor],
        prediction: typing.Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Evaluate the difference in the predicted and target dipole"""

        metric_func = get_metric(self.metric)
        target_dipole = labels[self.dipole_column]
        predicted_charges = prediction[self.charge_label]
        # get the conformation in bohr
        conformation = labels[self.conformation_column]
        predicted_dipole = torch.matmul(predicted_charges, conformation)
        return (
            metric_func(predicted_dipole, target_dipole)
            * self.weight
            / self.denominator
        )

    def target_column(self) -> str:
        return self.dipole_column


LossCalculator = typing.Union[typing.Literal["ReadoutTarget", "DipoleTarget"], str]


def get_loss_function(type_: LossCalculator) -> typing.Type[_BaseTarget]:
    """Get the loss calculator based on the type."""
    if type_.lower() == "readouttarget":
        return ReadoutTarget
    elif type_.lower() == "dipoletarget":
        return DipoleTarget
    else:
        raise NotImplementedError(f"Loss calculator {type_} not supported.")
