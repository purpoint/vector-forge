"""The pipeline: the engine's front door.

Phases 1-4 built four tools (chunker, embedder, similarity, store). This file
wires them together so the whole engine works through ONE object with two
verbs:

    ingest(text)     -> chunk -> embed -> wrap in Records -> add to store
    retrieve(question) -> embed question -> store.search -> SearchHits

Crucially, the pipeline talks only to the Embedder and VectorStore *interfaces*,
never to a specific implementation. Swap the offline embedder for a real model,
or the in-memory store for pgvector, and none of this code changes.
"""

from __future__ import annotations

import uuid

from .chunking import chunk_text
from .embeddings import Embedder, HashEmbedder
from .vectorstore import InMemoryVectorStore, Record, SearchHit, VectorStore


class RAGPipeline:
    def __init__(
        self,
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
        chunk_size: int = 800,
        overlap: int = 150,
        top_k: int = 4,
    ):
        # Defaults keep it runnable with zero setup; pass your own to swap parts.
        self.embedder = embedder or HashEmbedder()
        self.store = store or InMemoryVectorStore()
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.top_k = top_k

    def ingest(self, text: str, doc_id: str | None = None) -> int:
        """Learn a document. Returns how many chunks were stored."""
        doc_id = doc_id or f"doc-{uuid.uuid4().hex[:8]}"

        # 1. chunk
        chunks = chunk_text(text, doc_id=doc_id,
                            chunk_size=self.chunk_size, overlap=self.overlap)
        if not chunks:
            return 0

        # 2. embed every chunk in one batch
        vectors = self.embedder.embed([c.text for c in chunks])

        # 3. wrap each chunk in a Record (vector glued to its text)
        records = [
            Record(
                id=f"{doc_id}:{c.chunk_index}",
                text=c.text,
                doc_id=c.doc_id,
                chunk_index=c.chunk_index,
                metadata=c.metadata,
            )
            for c in chunks
        ]

        # 4. add to the store
        self.store.add(vectors, records)
        return len(chunks)

    def retrieve(self, question: str, k: int | None = None) -> list[SearchHit]:
        """Find the most relevant chunks for a question."""
        k = k or self.top_k
        query_vector = self.embedder.embed([question])[0]
        return self.store.search(query_vector, k)