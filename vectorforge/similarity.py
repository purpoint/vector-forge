"""Hand-written vector similarity — the heart of VectorForge.

No vector database does the work here. We compute cosine similarity ourselves
with numpy and pick the top-k matches by brute force. For a corpus that fits
in memory this is exact and fast. The scaling path (pgvector / an ANN index)
lives behind the VectorStore interface, so this code never has to change.

Cosine similarity measures the ANGLE between two vectors, ignoring length:
    cos(a, b) = (a . b) / (||a|| * ||b||)
Same direction -> 1.0, perpendicular -> 0.0, opposite -> -1.0.
Direction carries meaning; length is just how much text. So we compare direction.
"""

from __future__ import annotations

import numpy as np


def l2_normalize(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Scale each row to length 1.

    Once vectors are unit-length, cosine similarity becomes a plain dot
    product (the denominator is 1), so we normalize once and skip dividing
    by norms on every comparison. eps avoids dividing by zero on a zero vector.
    """
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix[np.newaxis, :]
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, eps)


def cosine_similarity(query: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    """Cosine similarity between one query vector and every corpus row.

    Returns a 1-D array of scores in [-1, 1], one per corpus row.
    """
    query = l2_normalize(query)      # shape (1, d)
    corpus = l2_normalize(corpus)    # shape (n, d)
    # (1, d) . (d, n) -> (1, n) -> (n,)
    return (query @ corpus.T).ravel()


def top_k(scores: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Return the (index, score) of the k highest scores, best first.

    Uses argpartition for an O(n) partial selection, then sorts only the k
    survivors. A full sort would be O(n log n) and we never need the tail
    ordered.
    """
    scores = np.asarray(scores)
    n = scores.shape[0]
    if n == 0:
        return []
    k = min(k, n)
    partitioned = np.argpartition(scores, n - k)[-k:]
    ordered = partitioned[np.argsort(scores[partitioned])[::-1]]
    return [(int(i), float(scores[i])) for i in ordered]