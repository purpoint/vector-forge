import numpy as np
from vectorforge.embeddings import HashEmbedder, Embedder


def test_embedder_returns_correct_shape():
    e = HashEmbedder(dim=32)
    vecs = e.embed(["hello world", "foo bar baz"])
    assert vecs.shape == (2, 32)


def test_empty_input_returns_empty_array():
    e = HashEmbedder(dim=32)
    vecs = e.embed([])
    assert vecs.shape == (0, 32)


def test_same_text_gives_same_vector():
    # Deterministic: embedding the same text twice must match exactly.
    e = HashEmbedder(dim=64)
    a = e.embed(["repeatable text"])
    b = e.embed(["repeatable text"])
    assert np.array_equal(a, b)


def test_shared_words_produce_overlap():
    e = HashEmbedder(dim=128)
    vecs = e.embed(["the cat sat", "the cat ran"])
    # Both share "the" and "cat", so their vectors must overlap somewhere.
    overlap = np.logical_and(vecs[0] > 0, vecs[1] > 0).sum()
    assert overlap > 0


def test_hashembedder_satisfies_protocol():
    # HashEmbedder honors the Embedder contract.
    assert isinstance(HashEmbedder(), Embedder)