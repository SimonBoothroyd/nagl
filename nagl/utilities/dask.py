from typing import TYPE_CHECKING

from nagl.utilities import requires_package

if TYPE_CHECKING:
    from dask_jobqueue import LSFCluster
    from distributed import LocalCluster


@requires_package("dask.distributed")
def setup_dask_local_cluster(n_workers: int) -> "LocalCluster":
    """Set up a dask cluster which parallelizes tasks over processes on a single machine.

    Args:
        n_workers: The number of workers to spawn.

    Returns:
        The initialized cluster.
    """
    from distributed import LocalCluster

    return LocalCluster(n_workers=n_workers)


@requires_package("dask_jobqueue")
def setup_dask_lsf_cluster(
    n_workers: int,
    queue: str,
    memory_gigabytes: int,
    wall_time: str,
    environment_name: str,
) -> "LSFCluster":
    """Set up a dask cluster which integrates with an existing LSF queue manager to
    spawn and manage workers.

    Args:
        n_workers: The number of workers to spawn.
        queue: The queue to submit the workers to.
        memory_gigabytes: The maximum memory to request per worker in GB.
        wall_time: The maximum wall-clock time to spawn each worker for.
        environment_name: The conda environment to activate for each worker.

    Returns:
        The initialized cluster.
    """
    from dask_jobqueue import LSFCluster

    cluster = LSFCluster(
        queue=queue,
        cores=1,
        memory=f"{memory_gigabytes * 1e9}B",
        walltime=wall_time,
        local_directory="dask-worker-space",
        log_directory="dask-worker-logs",
        env_extra=[f"conda activate {environment_name}"],
    )
    cluster.scale(n=n_workers)

    return cluster
