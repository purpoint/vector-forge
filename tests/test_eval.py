from eval.harness import evaluate, load_dataset
from vectorforge.pipeline import RAGPipeline
from pathlib import Path


def _loaded_pipeline():
    data = load_dataset(Path("eval/sample_qa.json"))
    rag = RAGPipeline()
    for doc in data["documents"]:
        rag.ingest(doc["text"], doc_id=doc["doc_id"])
    return rag, data["questions"]


def test_hit_rate_is_between_0_and_1():
    rag, questions = _loaded_pipeline()
    result = evaluate(rag, questions, k=3)
    assert 0.0 <= result.hit_rate <= 1.0
    assert 0.0 <= result.mrr <= 1.0


def test_hit_rate_improves_or_holds_with_larger_k():
    # finding the right doc can only get easier with a bigger k, never harder
    rag, questions = _loaded_pipeline()
    r1 = evaluate(rag, questions, k=1)
    r3 = evaluate(rag, questions, k=3)
    assert r3.hit_rate >= r1.hit_rate


def test_perfect_retrieval_scores_one():
    # a question that exactly matches a doc should be retrievable at k=1
    rag = RAGPipeline()
    rag.ingest("The capital of France is Paris.", doc_id="france")
    rag.ingest("Bananas are yellow fruit.", doc_id="banana")
    from eval.harness import evaluate
    q = [{"question": "What is the capital of France?", "relevant_doc_ids": ["france"]}]
    assert evaluate(rag, q, k=1).hit_rate == 1.0