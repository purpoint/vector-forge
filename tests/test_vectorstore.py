import numpy as np
import pytest
from vectorforge.embeddings import HashEmbedder
from vectorforge.vectorstore import (
    InMemoryVectorStore, PgVectorStore, Record, VectorStore,
)


def _make_store():
    e = HashEmbedder(dim=128)
    store = InMemoryVectorStore()
    docs = ["cats purr", "dogs bark", "python code"]
    vecs = e.embed(docs)
    recs = [Record(id=f"r{i}", text=d, doc_id=f"d{i}", chunk_index=0, metadata={})
            for i, d in enumerate(docs)]
    store.add(vecs, recs)
    return store, e


def test_add_increases_length():
    store, _ = _make_store()
    assert len(store) == 3


def test_search_returns_most_relevant_first():
    store, e = _make_store()
    q = e.embed(["cats"])[0]
    hits = store.search(q, k=2)
    assert hits[0].record.text == "cats purr"


def test_search_on_empty_store_returns_empty():
    store = InMemoryVectorStore()
    q = np.zeros(128)
    assert store.search(q, k=3) == []


def test_mismatched_lengths_raise():
    store = InMemoryVectorStore()
    with pytest.raises(ValueError):
        store.add(np.zeros((2, 4)), [Record("r", "t", "d", 0, {})])  # 2 vecs, 1 record


def test_save_and_load_roundtrip(tmp_path):
    store, e = _make_store()
    store.save(tmp_path)
    reloaded = InMemoryVectorStore.load(tmp_path)
    assert len(reloaded) == 3
    q = e.embed(["cats"])[0]
    assert reloaded.search(q, k=1)[0].record.text == "cats purr"


def test_pgvector_is_documented_stub():
    with pytest.raises(NotImplementedError):
        PgVectorStore()


def test_inmemory_satisfies_interface():
    assert issubclass(InMemoryVectorStore, VectorStore)