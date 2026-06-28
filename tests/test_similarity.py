import numpy as np
from vectorforge.similarity import l2_normalize, cosine_similarity, top_k


def test_normalize_gives_unit_length_rows():
    m = np.array([[3.0, 4.0]])          # length is 5
    out = l2_normalize(m)
    assert abs(np.linalg.norm(out[0]) - 1.0) < 1e-6


def test_identical_vectors_score_one():
    v = np.array([[1.0, 2.0, 3.0]])
    assert abs(cosine_similarity(v, v)[0] - 1.0) < 1e-6


def test_orthogonal_vectors_score_zero():
    q = np.array([1.0, 0.0])
    corpus = np.array([[0.0, 1.0]])
    assert abs(cosine_similarity(q, corpus)[0]) < 1e-6


def test_opposite_vectors_score_minus_one():
    q = np.array([1.0, 0.0])
    corpus = np.array([[-1.0, 0.0]])
    assert abs(cosine_similarity(q, corpus)[0] - (-1.0)) < 1e-6


def test_top_k_orders_by_score_descending():
    scores = np.array([0.1, 0.9, 0.5, 0.95, 0.3])
    result = top_k(scores, 3)
    assert [i for i, _ in result] == [3, 1, 2]


def test_top_k_handles_k_larger_than_n():
    scores = np.array([0.2, 0.8])
    assert len(top_k(scores, 10)) == 2


def test_zero_vector_does_not_crash():
    q = np.zeros(4)
    corpus = np.array([[0.0, 0.0, 0.0, 0.0]])
    assert np.isfinite(cosine_similarity(q, corpus)[0])