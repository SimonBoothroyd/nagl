from nagl.utilities.resonance._caching import PathCache


class TestPathCache:
    def test_init(self, nx_carboxylate):

        cache = PathCache(nx_carboxylate, max_path_length=5)
        assert cache._nx_graph == nx_carboxylate
        assert cache._max_path_length == 5

    def test_all_odd_n_simple_paths(self, nx_carboxylate):

        cache = PathCache(nx_carboxylate)

        assert cache.all_odd_n_simple_paths(1, 2) == ((1, 0, 2),)

        # remove the graph to make sure the method is actually cached.
        cache._nx_graph = None

        assert cache.all_odd_n_simple_paths(1, 2) == ((1, 0, 2),)
        assert cache.all_odd_n_simple_paths(2, 1) == ((2, 0, 1),)

    def test_all_odd_n_simple_paths_max_length(self, nx_carboxylate):

        cache = PathCache(nx_carboxylate, max_path_length=1)
        assert cache.all_odd_n_simple_paths(1, 2) == tuple()
