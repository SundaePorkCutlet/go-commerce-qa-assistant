import os
import subprocess
import sys

import numpy as np
from chromadb.errors import NotFoundError

from pqa.config import Settings
from pqa import vector_store
from pqa.vector_store import LocalHashEmbeddingFunction, _stable_token_index


def test_stable_token_index_is_independent_from_python_hash_seed() -> None:
    script = (
        "from pqa.vector_store import _stable_token_index; "
        "print(_stable_token_index('stock.reserved', 256))"
    )
    values: list[str] = []
    for seed in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        output = subprocess.check_output([sys.executable, "-c", script], env=env, text=True)
        values.append(output.strip())

    assert values[0] == values[1] == str(_stable_token_index("stock.reserved", 256))


def test_local_hash_embedding_maps_same_token_to_same_dimension() -> None:
    first = LocalHashEmbeddingFunction(dim=256)(["stock.reserved"])[0]
    second = LocalHashEmbeddingFunction(dim=256)(["stock.reserved"])[0]
    idx = _stable_token_index("stock.reserved", 256)

    assert np.array_equal(first, second)
    assert first[idx] > 0


def test_reset_collection_creates_collection_when_it_does_not_exist(monkeypatch) -> None:
    expected_collection = object()

    class MissingCollectionClient:
        def delete_collection(self, _name: str) -> None:
            raise NotFoundError("collection does not exist")

        def get_or_create_collection(self, **_kwargs):
            return expected_collection

    monkeypatch.setattr(vector_store, "_get_client", lambda _settings: MissingCollectionClient())

    assert vector_store.reset_collection(Settings()) is expected_collection
