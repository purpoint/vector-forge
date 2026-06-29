from vectorforge.pipeline import RAGPipeline


def test_ingest_returns_chunk_count():
    rag = RAGPipeline()
    n = rag.ingest("One sentence. Two sentence. Three sentence.", doc_id="d")
    assert n >= 1


def test_ingest_empty_text_stores_nothing():
    rag = RAGPipeline()
    assert rag.ingest("", doc_id="d") == 0
    assert len(rag.store) == 0


def test_retrieve_finds_relevant_chunk():
    rag = RAGPipeline()
    rag.ingest("The capital of France is Paris.", doc_id="geo")
    rag.ingest("Python is a programming language.", doc_id="prog")
    hits = rag.retrieve("What is the capital of France?", k=1)
    assert hits[0].record.doc_id == "geo"


def test_retrieve_respects_k():
    rag = RAGPipeline()
    rag.ingest("Alpha. Bravo. Charlie. Delta. Echo. Foxtrot.", doc_id="d",)
    hits = rag.retrieve("alpha", k=2)
    assert len(hits) <= 2


def test_pipeline_works_through_interfaces():
    # Passing custom parts proves the pipeline depends only on the contracts.
    from vectorforge.embeddings import HashEmbedder
    from vectorforge.vectorstore import InMemoryVectorStore
    rag = RAGPipeline(embedder=HashEmbedder(dim=64), store=InMemoryVectorStore())
    rag.ingest("swappable parts work", doc_id="d")
    assert len(rag.store) >= 1