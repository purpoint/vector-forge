"""Vector storage behind a single interface.

This is where our day-one scaling decision becomes real. The VectorStore
abstract class is a contract: every backend must support add() and search().
InMemoryVectorStore implements it with the hand-written cosine code from
similarity.py. A future PgVectorStore would implement the SAME contract with
Postgres + pgvector — and the rest of the engine would never notice the swap.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from .similarity import cosine_similarity, top_k


@dataclass
class Record:
    """A stored chunk: its text plus where it came from."""
    id: str
    text: str
    doc_id: str
    chunk_index: int
    metadata: dict


@dataclass
class SearchHit:
    """A search result: the matching record and its similarity score."""
    record: Record
    score: float


class VectorStore(ABC):
    """The contract every storage backend must follow."""

    @abstractmethod
    def add(self, vectors: np.ndarray, records: list[Record]) -> None:
        ...

    @abstractmethod
    def search(self, query_vector: np.ndarray, k: int) -> list[SearchHit]:
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...


class InMemoryVectorStore(VectorStore):
    """Exact cosine search over an in-memory numpy matrix.

    Holds all vectors in one (n, dim) array and all records in a parallel
    list. Search is a single normalize + matrix multiply, so a few tens of
    thousands of chunks return in milliseconds.
    """

    def __init__(self):
        self._vectors: np.ndarray | None = None
        self._records: list[Record] = []

    def add(self, vectors: np.ndarray, records: list[Record]) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.shape[0] != len(records):
            raise ValueError("vectors and records length mismatch")
        if self._vectors is None:
            self._vectors = vectors
        else:
            self._vectors = np.vstack([self._vectors, vectors])
        self._records.extend(records)

    def search(self, query_vector: np.ndarray, k: int) -> list[SearchHit]:
        if self._vectors is None or not self._records:
            return []
        scores = cosine_similarity(query_vector, self._vectors)
        return [
            SearchHit(record=self._records[i], score=s)
            for i, s in top_k(scores, k)
        ]

    def __len__(self) -> int:
        return len(self._records)

    # --- persistence ---
    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if self._vectors is not None:
            np.save(directory / "vectors.npy", self._vectors)
        with open(directory / "records.jsonl", "w") as f:
            for r in self._records:
                f.write(json.dumps(asdict(r)) + "\n")

    @classmethod
    def load(cls, directory: str | Path) -> "InMemoryVectorStore":
        directory = Path(directory)
        store = cls()
        vec_path = directory / "vectors.npy"
        if vec_path.exists():
            store._vectors = np.load(vec_path)
        rec_path = directory / "records.jsonl"
        if rec_path.exists():
            with open(rec_path) as f:
                store._records = [Record(**json.loads(line)) for line in f]
        return store


class PgVectorStore(VectorStore):
    """Scaling path: Postgres + pgvector. (Documented stub.)

    The whole point of the VectorStore interface is that this class can
    replace InMemoryVectorStore with zero changes upstream. When the corpus
    outgrows memory, you implement these three methods against Postgres:

        CREATE EXTENSION IF NOT EXISTS vector;
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY, doc_id TEXT, chunk_index INT,
            text TEXT, metadata JSONB, embedding VECTOR(256)
        );
        CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops);

    Search becomes one SQL query (<=> is cosine distance, so score = 1 - dist):

        SELECT *, 1 - (embedding <=> %(q)s) AS score
        FROM chunks ORDER BY embedding <=> %(q)s LIMIT %(k)s;

    Left unimplemented so the project runs with no database.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "PgVectorStore is the documented scaling path. "
            "Use InMemoryVectorStore for the from-scratch implementation."
        )

    def add(self, vectors, records): ...
    def search(self, query_vector, k): ...
    def __len__(self): return 0