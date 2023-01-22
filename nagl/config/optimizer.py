import typing

import pydantic


@pydantic.dataclasses.dataclass
class OptimizerConfig:
    """Configuration of a PyTorch optimizer."""

    type: typing.Literal["Adam"] = pydantic.Field(
        ..., description="The type of optimizer to use."
    )

    lr: float = pydantic.Field(..., description="The optimizer learning rate.")
