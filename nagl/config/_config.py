import pydantic

from nagl.config.data import DataConfig
from nagl.config.model import ModelConfig
from nagl.config.optimizer import OptimizerConfig


@pydantic.dataclasses.dataclass
class Config:
    """The main configuration for models, datasets and optimizers in NAGL."""

    model: ModelConfig = pydantic.Field(..., description="The model configuration.")
    data: DataConfig = pydantic.Field(
        ...,
        description="The data configuration, including definitions of the train, val, "
        "and test sets.",
    )
    optimizer: OptimizerConfig = pydantic.Field(
        ..., description="The optimizer configuration."
    )
