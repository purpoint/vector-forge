"""Grounded answer generation — the 'G' in RAG.

The whole point of RAG is faithfulness: the model must answer ONLY from the
chunks we retrieved, and tell us where each claim came from. We enforce that
with (1) a strict system prompt forbidding outside knowledge and mandating
[n] citations, and (2) numbered context blocks the model cites by index.

Two backends, behind one tiny contract:
* EchoLLM      - offline, no API key. Stitches retrieved chunks into a
                 grounded-looking answer with citation markers. Proves the
                 plumbing (context + citations) without a paid model.
* AnthropicLLM - real generation via the Claude Messages API.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .vectorstore import SearchHit

SYSTEM_PROMPT = (
    "You are a retrieval-grounded assistant. Answer the user's question using "
    "ONLY the numbered context passages provided. If the answer is not in the "
    "context, say you don't know. Cite every claim with its passage number in "
    "square brackets, e.g. [1] or [2][3]. Do not use outside knowledge."
)


@dataclass
class Answer:
    """A generated answer: the text, which passages it cited, and the context used."""
    text: str
    citations: list[int]
    context_used: list[SearchHit]


def build_context_block(hits: list[SearchHit]) -> str:
    """Render retrieved chunks as a numbered list the model can cite by index."""
    lines = []
    for i, hit in enumerate(hits, start=1):
        lines.append(f"[{i}] (source: {hit.record.doc_id})\n{hit.record.text}")
    return "\n\n".join(lines)


def _extract_citations(text: str, n_passages: int) -> list[int]:
    """Find [n] markers in the answer, keep only valid passage numbers."""
    found = {int(m) for m in re.findall(r"\[(\d+)\]", text)}
    return sorted(c for c in found if 1 <= c <= n_passages)


class EchoLLM:
    """Offline stand-in: builds a grounded answer from the chunks, no API key."""

    def generate(self, question: str, hits: list[SearchHit]) -> Answer:
        if not hits:
            return Answer("I don't know - no relevant context was found.", [], [])
        snippets = []
        for i, hit in enumerate(hits, start=1):
            snippet = hit.record.text.strip().replace("\n", " ")
            if len(snippet) > 160:
                snippet = snippet[:160].rsplit(" ", 1)[0] + "..."
            snippets.append(f"{snippet} [{i}]")
        text = "Based on the retrieved context: " + " ".join(snippets)
        return Answer(text, list(range(1, len(hits) + 1)), hits)


class AnthropicLLM:
    """Real generation via Claude. Requires `pip install anthropic` and ANTHROPIC_API_KEY."""

    def __init__(self, model: str = "claude-sonnet-4-6"):
        import anthropic  # imported lazily so the package runs without it
        self._client = anthropic.Anthropic()
        self.model = model

    def generate(self, question: str, hits: list[SearchHit]) -> Answer:
        if not hits:
            return Answer("I don't know - no relevant context was found.", [], [])
        context = build_context_block(hits)
        user = f"Context passages:\n\n{context}\n\nQuestion: {question}"
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(block.text for block in msg.content if block.type == "text")
        citations = _extract_citations(text, len(hits))
        return Answer(text, citations, hits)


def build_llm(provider: str = "echo", **kwargs):
    """Factory: 'echo' (offline) or 'anthropic' (real)."""
    provider = provider.lower()
    if provider == "echo":
        return EchoLLM()
    if provider == "anthropic":
        return AnthropicLLM(**kwargs)
    raise ValueError(f"unknown llm provider: {provider!r}")