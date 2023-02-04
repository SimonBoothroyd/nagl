import pytest
import torch

from nagl.training import metrics


def test_rmse():
    actual = metrics.rmse(
        torch.tensor([0.0, 3.0]),
        torch.tensor([0.0, 4.0]),
    )
    expected = torch.sqrt(torch.tensor(0.5))  # (0.0 + 1.0) / 2

    assert torch.isclose(actual, expected)


@pytest.mark.parametrize(
    "type_, expected_func",
    [("rmse", metrics.rmse), ("mse", metrics.mse), ("mae", metrics.mae)],
)
def test_get_metric(type_, expected_func):
    func = metrics.get_metric(type_)
    assert callable(func)

    assert func == expected_func
