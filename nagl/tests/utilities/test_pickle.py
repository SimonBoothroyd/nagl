import os
import pickle

from nagl.utilities.pickle import unpickle_all


def test_unpickle_all(tmpdir):

    expected_list = [str(i) for i in range(10)]

    with open(os.path.join(tmpdir, "tmp.pkl"), "wb") as file:
        [pickle.dump(str(i), file) for i in expected_list]

    unpickled_list = [*unpickle_all(os.path.join(tmpdir, "tmp.pkl"))]

    assert unpickled_list == expected_list
