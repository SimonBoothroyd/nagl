import pickle
from typing import Any, Generator


def unpickle_all(file_path: str) -> Generator[Any, None, None]:
    """Returns a generator which iterates over each item in a multi-object pickle file.

    Args
        file_path: The path to the pickled file.

    Returns
        A generator over the items in the pickled file.
    """

    with open(file_path, "rb") as file:

        while True:
            try:
                yield pickle.load(file)
            except EOFError:
                break
