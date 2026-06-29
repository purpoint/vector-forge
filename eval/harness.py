"""Evaluation harness: how good is retrieval?

Generation quality is hard to score automatically, but retrieval is not:
for each question we know which document holds the answer, so we can ask
'did the right chunk land in the top-k?' If retrieval misses, no LLM can
save the answer -- so this is the most honest signal that the engine works.

Metrics:
* hit_rate@k : fraction of questions whose relevant doc appears in top-k.
* MRR        : mean reciprocal rank -- rewards ranking the right chunk higher.

Run:  python -m eval.harness
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from vectorforge.pipeline import RAGPipeline


@dataclass
class EvalResult:
    k: int
    n_questions: int
    hit_rate: float
    mrr: float


def evaluate(pipeline: RAGPipeline, questions: list[dict], k: int) -> EvalResult:
    hits = 0
    reciprocal_ranks = 0.0
    for item in questions:
        relevant = set(item["relevant_doc_ids"])
        retrieved = pipeline.retrieve(item["question"], k=k)
        # find the rank (1-based) of the first relevant chunk, if any
        rank = None
        for i, hit in enumerate(retrieved, start=1):
            if hit.record.doc_id in relevant:
                rank = i
                break
        if rank is not None:
            hits += 1
            reciprocal_ranks += 1.0 / rank
    n = len(questions)
    return EvalResult(
        k=k,
        n_questions=n,
        hit_rate=hits / n if n else 0.0,
        mrr=reciprocal_ranks / n if n else 0.0,
    )


def load_dataset(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def main() -> None:
    data = load_dataset(Path(__file__).parent / "sample_qa.json")

    pipeline = RAGPipeline()
    for doc in data["documents"]:
        pipeline.ingest(doc["text"], doc_id=doc["doc_id"])

    print(f"Indexed {len(pipeline.store)} chunks "
          f"from {len(data['documents'])} documents.\n")
    print(f"{'k':>3} | {'hit_rate':>9} | {'MRR':>6}")
    print("-" * 26)
    for k in (1, 3, 5):
        r = evaluate(pipeline, data["questions"], k=k)
        print(f"{r.k:>3} | {r.hit_rate:>9.3f} | {r.mrr:>6.3f}")
    print(f"\n(evaluated on {len(data['questions'])} questions)")


if __name__ == "__main__":
    main()