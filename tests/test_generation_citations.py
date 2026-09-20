from uuid import uuid4

import pytest

from app.domain.generation.citations import (
    build_sources,
    extract_citations,
    system_prompt,
    user_prompt,
)
from app.domain.retrieval.entities import RetrievedChunk


def _hit(
    *,
    title: str = "Guide",
    content: str = "some documentation content",
    page: int = 2,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid4(),
        document_id=uuid4(),
        content=content,
        chunk_index=0,
        page_number=page,
        metadata={"heading": "Intro"},
        document_title=title,
    )


def test_build_sources_numbers_and_annotates_chunks() -> None:
    first = _hit(content="alpha")
    second = _hit(content="beta", title="Reference")

    sources, passages = build_sources([first, second])

    assert sources.startswith("[1] Guide, page 2\nalpha")
    assert "[2] Reference, page 2\nbeta" in sources
    assert [
        passage.chunk is hit
        for passage, hit in zip(passages, [first, second], strict=True)
    ]
    assert [passage.excerpt for passage in passages] == ["alpha", "beta"]


def test_build_sources_truncates_long_passages() -> None:
    long_chunk = _hit(content="x" * 500)

    _sources, passages = build_sources([long_chunk], max_excerpt_chars=100)

    assert passages[0].excerpt == ("x" * 100) + "..."
    assert len(passages[0].excerpt) == 103


def test_build_sources_respects_total_budget() -> None:
    chunks = [_hit(content="a" * 20), _hit(content="b" * 20), _hit(content="c" * 20)]

    sources, passages = build_sources(chunks, max_total_chars=100)

    assert len(passages) == 2
    assert "[3]" not in sources
    assert chunks[0] is passages[0].chunk
    assert chunks[1] is passages[1].chunk


def test_build_sources_includes_first_passage_even_over_budget() -> None:
    chunks = [_hit(content="x" * 500)]

    _sources, passages = build_sources(chunks, max_total_chars=50)

    assert len(passages) == 1
    assert passages[0].chunk is chunks[0]


def test_build_sources_empty_chunks() -> None:
    sources, passages = build_sources([])

    assert sources == ""
    assert passages == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_excerpt_chars": 0},
        {"max_total_chars": 0},
    ],
)
def test_build_sources_rejects_non_positive_limits(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        build_sources([_hit()], **kwargs)


def test_user_prompt_embeds_question_and_sources() -> None:
    prompt = user_prompt("How does it work?", "[1] Guide, page 2\ncontent")

    assert prompt == (
        "Question: How does it work?\n\nDocumentation:\n"
        "[1] Guide, page 2\ncontent"
    )


def test_user_prompt_without_sources_tells_model_not_to_invent() -> None:
    prompt = user_prompt("How does it work?", "")

    assert prompt.startswith("Question: How does it work?")
    assert "No relevant documentation was found" in prompt


def test_system_prompt_grounds_the_model() -> None:
    prompt = system_prompt()

    assert "ONLY the documentation excerpts" in prompt
    assert "[1]" in prompt


def test_extract_citations_maps_numbers_to_chunks_in_order() -> None:
    chunks = [_hit(content="alpha"), _hit(content="beta"), _hit(content="gamma")]
    _sources, passages = build_sources(chunks)

    citations = extract_citations(
        "The beta method [2] and gamma step [3] and beta again [2].", passages
    )

    assert [citation.chunk_id for citation in citations] == [
        chunks[1].chunk_id,
        chunks[2].chunk_id,
    ]
    assert citations[0].document_title == "Guide"
    assert citations[0].page_number == 2
    assert citations[0].excerpt == "beta"
    assert citations[0].metadata == {"heading": "Intro"}


def test_extract_citations_ignores_out_of_range_numbers() -> None:
    chunks = [_hit(content="alpha")]
    _sources, passages = build_sources(chunks)

    citations = extract_citations("Alpha [1] and nothing [9].", passages)

    assert [citation.chunk_id for citation in citations] == [chunks[0].chunk_id]


def test_extract_citations_returns_empty_without_references() -> None:
    chunks = [_hit(content="alpha")]
    _sources, passages = build_sources(chunks)

    assert extract_citations("No citations here.", passages) == []
    assert extract_citations("", passages) == []