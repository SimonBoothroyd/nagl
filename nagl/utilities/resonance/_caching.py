import typing

import networkx

if typing.TYPE_CHECKING:
    pass


class PathCache:
    """A utility class that caches finding all simple paths between sets of nodes in
    a ``networkx`` graph."""

    def __init__(
        self, nx_graph: networkx.Graph, max_path_length: typing.Optional[int] = None
    ):
        """

        Args:
            nx_graph: The graph to path find on.
            max_path_length: The maximum path length to find.
        """

        self._nx_graph = nx_graph
        self._max_path_length = max_path_length

        self._cache: typing.Dict[
            typing.Tuple[int, int], typing.Tuple[typing.Tuple[int, ...]]
        ] = {}

    def all_odd_n_simple_paths(
        self, index_a: int, index_b: int
    ) -> typing.Tuple[typing.Tuple[int, ...]]:
        """Generate all simple paths in the graph G from ``index_a`` to ``index_b`` that
        contain an **odd** number of nodes.
        """

        if (index_a, index_b) in self._cache:
            return self._cache[(index_a, index_b)]
        elif (index_b, index_a) in self._cache:
            return tuple(path[::-1] for path in self._cache[(index_b, index_a)])

        # filter out any paths with even lengths as these cannot transfer paths
        path_generator: typing.Generator[int, None, None]
        paths = []

        for path_generator in networkx.all_simple_paths(
            self._nx_graph, index_a, index_b, cutoff=self._max_path_length
        ):
            path = tuple(path_generator)

            if len(path) == 0 or len(path) % 2 == 0:
                continue

            paths.append(path)

        self._cache[(index_a, index_b)] = tuple(paths)
        return self._cache[(index_a, index_b)]
