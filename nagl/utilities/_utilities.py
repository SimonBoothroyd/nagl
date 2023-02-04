import contextlib
import multiprocessing
import typing

_S = typing.TypeVar("_S")
_T = typing.TypeVar("_T")


@contextlib.contextmanager
def get_map_func(
    n_processes: int,
) -> typing.Callable[
    [typing.Callable[[_S], _T], typing.Iterable[_S]], typing.Iterable[_T]
]:
    """Returns either an interactive parallel map or a standard map depending on the
    number of processes

    Args:
        n_processes: The number of processes to distribute the mapping over. If 0
        the standard ``map`` function will be yielded, otherwise the
        ``multiprocessing.Pool.imap`` function will be.
    """

    if n_processes > 0:

        with multiprocessing.Pool(n_processes) as pool:
            yield pool.imap

    else:
        yield map
