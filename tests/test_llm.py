from vectorforge.llm import (
    EchoLLM, build_context_block, _extract_citations, build_llm, Answer,
)
from vectorforge.vectorstore import Record, SearchHit


def _hits():
    return [
        SearchHit(Record("r1", "Plants release oxygen.", "bio", 0, {}), 0.9),
        SearchHit(Record("r2", "Chlorophyll absorbs sunlight.", "bio", 1, {}), 0.7),
    ]


def test_context_block_is_numbered():
    block = build_context_block(_hits())
    assert "[1]" in block and "[2]" in block


def test_extract_citations_keeps_valid_only():
    # [3] is out of range when only 2 passages exist; it should be dropped.
    assert _extract_citations("facts [1] and [2] and [3]", n_passages=2) == [1, 2]


def test_echo_llm_produces_answer_with_citations():
    ans = EchoLLM().generate("why?", _hits())
    assert isinstance(ans, Answer)
    assert ans.citations == [1, 2]
    assert "[1]" in ans.text


def test_echo_llm_says_dont_know_with_no_context():
    ans = EchoLLM().generate("why?", [])
    assert ans.citations == []
    assert "don't know" in ans.text.lower()


def test_build_llm_factory():
    assert isinstance(build_llm("echo"), EchoLLM)


def test_pipeline_query_end_to_end():
    from vectorforge.pipeline import RAGPipeline
    rag = RAGPipeline()
    rag.ingest("The Eiffel Tower is 330 metres tall.", doc_id="eiffel")
    ans = rag.query("How tall is the Eiffel Tower?")
    assert ans.context_used          # something was retrieved
    assert ans.citations             # and cited