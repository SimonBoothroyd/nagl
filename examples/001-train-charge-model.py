import logging
import pathlib
import typing

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import MLFlowLogger

from nagl.config import Config, DataConfig, ModelConfig, OptimizerConfig
from nagl.config.data import Dataset, Target
from nagl.config.model import GCNConvolutionModule, ReadoutModule, Sequential
from nagl.features import AtomConnectivity, AtomFeature, AtomicElement, BondFeature
from nagl.training import DGLMoleculeDataModule, DGLMoleculeLightningModel


def configure_model(
    atom_features: typing.List[AtomFeature],
    bond_features: typing.List[BondFeature],
    n_gcn_layers: int,
    n_gcn_hidden_features: int,
    n_am1_layers: int,
    n_am1_hidden_features: int,
) -> ModelConfig:

    return ModelConfig(
        atom_features=atom_features,
        bond_features=bond_features,
        convolution=GCNConvolutionModule(
            type="SAGEConv",
            hidden_feats=[n_gcn_hidden_features] * n_gcn_layers,
            activation=["ReLU"] * n_gcn_layers,
        ),
        readouts={
            "charges": ReadoutModule(
                pooling="atom",
                forward=Sequential(
                    hidden_feats=[n_am1_hidden_features] * n_am1_layers + [2],
                    activation=["ReLU"] * n_am1_layers + ["Identity"],
                ),
                postprocess="charges",
            )
        },
    )


def configure_data() -> DataConfig:

    return DataConfig(
        training=Dataset(
            sources=["000-label-data/train.parquet"],
            # The 'column' must match one of the label columns in the parquet
            # table that was create during stage 000.
            # The 'readout' column should correspond to one our or model readout
            # keys.
            targets=[Target(column="charges-am1bcc", readout="charges", metric="rmse")],
        ),
        validation=Dataset(
            sources=["000-label-data/val.parquet"],
            targets=[Target(column="charges-am1bcc", readout="charges", metric="rmse")],
        ),
        test=Dataset(
            sources=["000-label-data/test.parquet"],
            targets=[Target(column="charges-am1bcc", readout="charges", metric="rmse")],
        ),
    )


def configure_optimizer(lr: float) -> OptimizerConfig:
    return OptimizerConfig(type="Adam", lr=lr)


def main():

    logging.basicConfig(level=logging.INFO)
    output_dir = pathlib.Path("001-train-charge-model")

    # Configure our model, data sets, and optimizer.
    model_config = configure_model(
        atom_features=[AtomicElement(values=["C", "H"]), AtomConnectivity()],
        bond_features=[],
        n_gcn_layers=4,
        n_gcn_hidden_features=128,
        n_am1_layers=3,
        n_am1_hidden_features=64,
    )
    data_config = configure_data()

    optimizer_config = configure_optimizer(0.001)

    # Define the model and lightning data module that will contain the train, val,
    # and test dataloaders if specified in ``data_config``.
    config = Config(model=model_config, data=data_config, optimizer=optimizer_config)

    model = DGLMoleculeLightningModel(config)
    print("Model", model)

    # The 'cache_dir' will store the fully featurized molecules so we don't need to
    # re-compute these each to we adjust a hyperparameter for example.
    data = DGLMoleculeDataModule(config, cache_dir=output_dir / "feature-cache")

    # Define an MLFlow experiment to store the outputs of training this model. This
    # Will include the usual statistics as well as useful artifacts highlighting
    # the models weak spots.
    logger = MLFlowLogger(
        experiment_name="am1bcc-charge-model", save_dir=str(output_dir / "mlruns")
    )

    # The MLFlow UI can be opened by running:
    #
    #    mlflow ui --backend-store-uri     ./001-train-charge-model/mlruns \
    #              --default-artifact-root ./001-train-charge-model/mlruns
    #

    # Train the model
    n_epochs = 100

    n_gpus = 0 if not torch.cuda.is_available() else 1
    print(f"Using {n_gpus} GPUs")

    trainer = pl.Trainer(
        gpus=n_gpus,
        min_epochs=n_epochs,
        max_epochs=n_epochs,
        logger=logger,
        log_every_n_steps=1,
    )

    trainer.fit(model, datamodule=data)
    trainer.test(model, datamodule=data)


if __name__ == "__main__":
    main()
