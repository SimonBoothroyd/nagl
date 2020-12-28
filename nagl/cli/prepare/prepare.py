import click

from nagl.cli.prepare.enumerate import enumerate_cli
from nagl.cli.prepare.filter import filter_cli


@click.group(
    "prepare",
    short_help="CLIs for preparing molecule sets.",
    help="CLIs for preparing molecule sets, such as filtering out molecules which are "
    "too large or contain unwanted chemistries, removing counter-ions, or enumerating "
    "possible tautomers / protomers.",
)
def prepare_cli():
    pass


prepare_cli.add_command(filter_cli)
prepare_cli.add_command(enumerate_cli)
