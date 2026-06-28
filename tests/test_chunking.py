from vectorforge.chunking import chunk_text


def test_empty_text_yields_no_chunks():
    assert chunk_text("", doc_id="d") == []


def test_single_short_text_is_one_chunk():
    chunks = chunk_text("One simple sentence here.", doc_id="d")
    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].doc_id == "d"


def test_long_text_splits_into_multiple_chunks():
    text = " ".join(f"Sentence number {i} sits here." for i in range(40))
    chunks = chunk_text(text, doc_id="d", chunk_size=120, overlap=30)
    assert len(chunks) > 1
    # Indices are sequential: 0, 1, 2, ...
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_chunks_respect_size_budget():
    text = " ".join(f"Sentence number {i} sits here." for i in range(40))
    chunks = chunk_text(text, doc_id="d", chunk_size=120, overlap=30)
    # Each chunk stays near the budget (size + one overlap's slack).
    for c in chunks:
        assert len(c.text) <= 120 + 30


def test_overlap_must_be_smaller_than_size():
    try:
        chunk_text("x", doc_id="d", chunk_size=50, overlap=50)
        assert False, "should have raised"
    except ValueError:
        pass