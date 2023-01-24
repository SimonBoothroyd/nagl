import typing

import pydantic

from nagl.features import AtomFeatureType, BondFeatureType

ActivationFunction = typing.Union[
    typing.Literal["Identity", "Tanh", "ReLU", "LeakyReLU", "SELU", "ELU"], str
]

PoolingType = typing.Literal["atom", "bond"]
PostprocessType = typing.Literal["charges"]


@pydantic.dataclasses.dataclass(config={"extra": pydantic.Extra.forbid})
class GCNConvolutionModule:
    """Configuration for a stack of GCN layers."""

    type: typing.Literal["SAGEConv"] = pydantic.Field(
        ..., description="The GCN architecture to use."
    )

    hidden_feats: typing.List[int] = pydantic.Field(
        ..., description="The number of hidden features to use in each layer."
    )
    activation: typing.List[ActivationFunction] = pydantic.Field(
        ..., description="The activation function to apply after each layer."
    )
    dropout: typing.Optional[typing.List[typing.Optional[float]]] = pydantic.Field(
        None, description="The dropout to apply after each layer."
    )


ConvolutionModuleType = typing.Union[GCNConvolutionModule]


@pydantic.dataclasses.dataclass(config={"extra": pydantic.Extra.forbid})
class Sequential:
    """Configuration for a sequential feed forward layer."""

    hidden_feats: typing.List[int] = pydantic.Field(
        ..., description="The number of hidden features to use in each layer."
    )
    activation: typing.List[ActivationFunction] = pydantic.Field(
        ..., description="The activation function to apply after each layer."
    )
    dropout: typing.Optional[typing.List[typing.Optional[float]]] = pydantic.Field(
        None, description="The dropout to apply after each layer."
    )


@pydantic.dataclasses.dataclass(config={"extra": pydantic.Extra.forbid})
class ReadoutModule:
    """Configuration for a readout layer, including the type of pooling, and
    postprocessing to perform."""

    pooling: PoolingType = pydantic.Field(
        ..., description="The type of Janossy pooling to apply."
    )
    forward: Sequential = pydantic.Field(
        ...,
        description="The feed forward network to map GCN features to desired outputs.",
    )
    postprocess: typing.Optional[PostprocessType] = pydantic.Field(
        ..., description="An extra layer to pass the feed forward outputs through."
    )


@pydantic.dataclasses.dataclass
class ModelConfig:
    """Configuration of the convolutional model with multiple readouts."""

    atom_features: typing.List[AtomFeatureType] = pydantic.Field(
        ..., description="The atom features to use."
    )
    bond_features: typing.List[BondFeatureType] = pydantic.Field(
        [],
        description="The bond features to use *if* the GCN architecture supports them.",
    )

    convolution: ConvolutionModuleType = pydantic.Field(
        ..., description="The convolution module to pass the molecule features through."
    )
    readouts: typing.Dict[str, ReadoutModule] = pydantic.Field(
        ..., description="The readout nn to map convolution features to outputs."
    )
