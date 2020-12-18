import click

from nagl.cli.label import label


@click.group()
def cli():
    """The root group for all CLI commands."""


cli.add_command(label)
