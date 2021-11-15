import os

from openff.utilities import temporary_cd

from nagl.utilities.dask import setup_dask_local_cluster, setup_dask_lsf_cluster


def test_setup_dask_local_cluster(tmpdir):

    with temporary_cd(str(tmpdir)):

        cluster = setup_dask_local_cluster(1)

        assert cluster is not None
        assert os.path.isdir("dask-worker-space")


def test_setup_dask_lsf_cluster(tmpdir):

    with temporary_cd(str(tmpdir)):

        cluster = setup_dask_lsf_cluster(1, "tmp", 1, "00:00", "tmp")
        assert cluster is not None
