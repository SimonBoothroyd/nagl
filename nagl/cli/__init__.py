import click

from nagl.cli.label import label_cli
from nagl.cli.prepare import prepare_cli


@click.group()
def cli():
    """A framework for learning classical force field parameters using graph
    convolutional neural networks.
    """


cli.add_command(prepare_cli)
cli.add_command(label_cli)

__all__ = ["cli"]
