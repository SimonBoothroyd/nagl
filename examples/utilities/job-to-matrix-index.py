import click
import numpy


@click.argument("n_options", nargs=-1, type=click.INT)
@click.argument("job_index", type=click.INT)
@click.command()
def main(n_options, job_index):

    matrix_indices = numpy.unravel_index(job_index - 1, n_options)
    print(" ".join(str(i) for i in matrix_indices))


if __name__ == "__main__":
    main()
