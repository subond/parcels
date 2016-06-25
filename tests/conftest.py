"""Global test configuration."""
import pytest  # NOQA
from subprocess import check_call
from mpi4py import MPI


def parallel(item):
    """Run a test in parallel.

    :arg item: The test item to run.
    """
    if MPI.COMM_WORLD.size > 1:
        raise RuntimeError("parallel test can't be run within parallel environment")
    marker = item.get_marker("parallel")
    if marker is None:
        raise RuntimeError("Parallel test doesn't have parallel marker")
    nprocs = marker.kwargs.get("nprocs", 3)
    if nprocs < 2:
        raise RuntimeError("Need at least two processes to run parallel test")

    # Only spew tracebacks on rank 0.
    # Run xfailing tests to ensure that errors are reported to calling process
    zerocall = " ".join(["py.test", "-s", "-q", str(item.fspath), "-k", item.name])
    restcall = " ".join(["py.test", "-s", "--tb=no", "-q", str(item.fspath), "-k", item.name])
    call = "mpiexec -n 1 %s : -n %d %s" % (zerocall, nprocs - 1, restcall)
    check_call(call, shell=True)


def pytest_configure(config):
    """Register an additional marker."""
    config.addinivalue_line(
        "markers",
        "parallel(nprocs): mark test to run in parallel on nprocs processors")


def pytest_runtest_setup(item):
    if item.get_marker("parallel"):
        if MPI.COMM_WORLD.size <= 1:
            # Blow away function arg in "master" process, to ensure
            # this test isn't run on only one process.
            item.obj = lambda *args, **kwargs: True


def pytest_runtest_call(item):
    if item.get_marker("parallel") and MPI.COMM_WORLD.size == 1:
        # Spawn parallel processes to run test
        parallel(item)
