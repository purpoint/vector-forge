"""Document chunking with overlap.

Embedding a whole document gives one blurry vector that mixes every topic.
So we split documents into smaller, overlapping windows. Each chunk carries
one focused idea (sharp vector), and the overlap keeps a thought that
straddles a boundary retrievable from at least one side.

Sizes are in characters for zero dependencies. A production system would
measure in tokens; only the length function would change, not the logic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Split on sentence-ending punctuation (. ! ?) followed by whitespace.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    """One piece of a document, plus where it came from."""
    text: str
    doc_id: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


def _split_sentences(text: str) -> list[str]:
    parts = _SENTENCE_RE.split(text.strip())
    return [p for p in parts if p]


def chunk_text(
    text: str,
    doc_id: str,
    chunk_size: int = 800,
    overlap: int = 150,
    metadata: dict | None = None,
) -> list[Chunk]:
    """Pack whole sentences into ~chunk_size windows, with char overlap.

    We add sentences to the current window until the next one would exceed
    chunk_size. Then we emit the window and start the next one by carrying
    back the trailing sentences worth ~overlap characters, so context flows
    across the seam.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    metadata = metadata or {}
    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: list[Chunk] = []
    window: list[str] = []
    window_len = 0
    chunk_index = 0

    def flush() -> None:
        nonlocal chunk_index
        body = " ".join(window).strip()
        if not body:
            return
        chunks.append(
            Chunk(
                text=body,
                doc_id=doc_id,
                chunk_index=chunk_index,
                metadata=dict(metadata),
            )
        )
        chunk_index += 1

    for sentence in sentences:
        addition = len(sentence) + (1 if window else 0)
        if window and window_len + addition > chunk_size:
            flush()
            # Build the overlap tail: keep trailing sentences until we've
            # carried back ~overlap characters.
            tail: list[str] = []
            tail_len = 0
            for s in reversed(window):
                if tail_len + len(s) > overlap:
                    break
                tail.insert(0, s)
                tail_len += len(s) + 1
            window = tail
            window_len = sum(len(s) + 1 for s in window)
        window.append(sentence)
        window_len += addition

    flush()
    return chunks