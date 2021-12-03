import tarfile

import click
from openeye import oechem
from tqdm import tqdm


@click.command()
@click.option(
    "--input",
    "input_path",
    help="The path to the input tarball of SDF files.",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option(
    "--output",
    "output_path",
    help="The path to save the output SDF file to.",
    type=click.Path(exists=False, file_okay=True, dir_okay=False),
)
def tar_to_sdf(input_path, output_path):
    """A CLI utility that will convert a tarball of SDF files into a (possibly
    compressed) single SDF file ."""

    output_stream = oechem.oemolostream(output_path)

    with tarfile.open(input_path) as tar_file:

        for member in tqdm(tar_file):

            sdf_contents = tar_file.extractfile(member).read()

            input_stream = oechem.oemolistream()
            input_stream.SetFormat(oechem.OEFormat_SDF)
            input_stream.openstring(sdf_contents)

            for oe_molecule in input_stream.GetOEMols():
                oechem.OEWriteMolecule(output_stream, oe_molecule)


if __name__ == "__main__":
    tar_to_sdf()
