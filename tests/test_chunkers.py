from pathlib import Path

import pytest

from app.domain.documents.entities import Document, DocumentPage
from app.infrastructure.ingestion.markdown_parser import MarkdownParser
from app.infrastructure.ingestion.section_chunker import SectionChunker
from app.infrastructure.ingestion.token_chunker import TokenChunker


def _document(pages: list[DocumentPage]) -> Document:
    return Document(title="Guide", source="test", pages=pages)


def _page(
    number: int,
    content: str,
    metadata: dict[str, str] | None = None,
) -> DocumentPage:
    return DocumentPage(
        page_number=number,
        content=content,
        metadata=metadata or {},
    )


class TestTokenChunker:
    def test_short_page_becomes_single_chunk(self) -> None:
        chunker = TokenChunker(token_limit=100)
        document = _document([_page(1, "alpha beta gamma", {"heading": "Intro"})])

        chunks = chunker.chunk(document)

        assert len(chunks) == 1
        assert chunks[0].content == "alpha beta gamma"
        assert chunks[0].page_number == 1
        assert chunks[0].document_id == document.id
        assert chunks[0].chunk_index == 0

    def test_windows_respect_token_limit(self) -> None:
        chunker = TokenChunker(token_limit=2)
        document = _document([_page(1, "a b c d e f")])

        chunks = chunker.chunk(document)

        assert [chunk.content for chunk in chunks] == [
            "a b",
            "c d",
            "e f",
        ]
        assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]

    def test_overlap_reuses_tokens_between_windows(self) -> None:
        chunker = TokenChunker(token_limit=4, token_overlap=2)
        document = _document([_page(1, "a b c d e f")])

        chunks = chunker.chunk(document)

        assert [chunk.content for chunk in chunks] == ["a b c d", "c d e f"]

    def test_overlap_prevents_trailing_duplicate_window(self) -> None:
        chunker = TokenChunker(token_limit=4, token_overlap=1)
        document = _document([_page(1, "t0 t1 t2 t3 t4 t5 t6 t7 t8 t9")])

        chunks = chunker.chunk(document)

        assert len(chunks) == 3
        assert [chunk.content for chunk in chunks] == [
            "t0 t1 t2 t3",
            "t3 t4 t5 t6",
            "t6 t7 t8 t9",
        ]

    def test_zero_overlap_does_not_reuse_tokens(self) -> None:
        chunker = TokenChunker(token_limit=2)
        document = _document([_page(1, "a b c d")])

        chunks = chunker.chunk(document)

        assert [chunk.content for chunk in chunks] == ["a b", "c d"]

    def test_chunk_index_is_continuous_across_pages(self) -> None:
        chunker = TokenChunker(token_limit=2)
        document = _document([_page(1, "a b c d"), _page(2, "e f g h")])

        chunks = chunker.chunk(document)

        assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2, 3]

    def test_each_chunk_keeps_its_own_page_number(self) -> None:
        chunker = TokenChunker(token_limit=2)
        document = _document([_page(1, "a b c d"), _page(2, "e f g h")])

        chunks = chunker.chunk(document)

        assert [chunk.page_number for chunk in chunks] == [1, 1, 2, 2]

    def test_page_metadata_propagates_into_every_chunk(self) -> None:
        chunker = TokenChunker(token_limit=2)
        document = _document([_page(1, "a b c d", {"heading": "Intro"})])

        chunks = chunker.chunk(document)

        assert [chunk.metadata for chunk in chunks] == [
            {"heading": "Intro"},
            {"heading": "Intro"},
        ]

    def test_empty_pages_are_skipped(self) -> None:
        chunker = TokenChunker(token_limit=2)
        document = _document([_page(1, ""), _page(2, "a b c d")])

        chunks = chunker.chunk(document)

        assert len(chunks) == 2
        assert [chunk.page_number for chunk in chunks] == [2, 2]

    @pytest.mark.parametrize(
        ("token_limit", "token_overlap"),
        [
            (0, 0),
            (-1, 0),
            (10, 10),
            (10, 20),
            (10, -1),
        ],
    )
    def test_invalid_config_raises(
        self,
        token_limit: int,
        token_overlap: int,
    ) -> None:
        with pytest.raises(ValueError):
            TokenChunker(token_limit=token_limit, token_overlap=token_overlap)


class TestSectionChunker:
    def test_merges_small_pages_into_one_chunk(self) -> None:
        chunker = SectionChunker(token_limit=100)
        document = _document(
            [
                _page(1, "alpha beta", {"heading": "A"}),
                _page(2, "gamma delta", {"heading": "B"}),
            ]
        )

        chunks = chunker.chunk(document)

        assert len(chunks) == 1
        assert chunks[0].content == "alpha beta\n\ngamma delta"
        assert chunks[0].page_number == 1
        assert chunks[0].chunk_index == 0
        assert chunks[0].metadata == {"heading": "A"}

    def test_flushes_group_when_next_page_exceeds_budget(self) -> None:
        chunker = SectionChunker(token_limit=3)
        document = _document(
            [_page(1, "a b", {"heading": "A"}), _page(2, "c d", {"heading": "B"})]
        )

        chunks = chunker.chunk(document)

        assert len(chunks) == 2
        assert chunks[0].content == "a b"
        assert chunks[0].page_number == 1
        assert chunks[0].metadata == {"heading": "A"}
        assert chunks[1].content == "c d"
        assert chunks[1].page_number == 2
        assert chunks[1].metadata == {"heading": "B"}

    def test_oversized_page_is_split_on_paragraph_boundaries(self) -> None:
        chunker = SectionChunker(token_limit=4)
        document = _document(
            [
                _page(
                    1,
                    "Heading\n\np1 alpha\n\np2 beta\n\np3 gamma",
                    {"heading": "Heading"},
                )
            ]
        )

        chunks = chunker.chunk(document)

        assert len(chunks) == 2
        assert chunks[0].content == "Heading\n\np1 alpha"
        assert chunks[1].content == "Heading\np2 beta\n\np3 gamma"
        assert chunks[0].metadata == {"heading": "Heading"}
        assert chunks[1].metadata == {"heading": "Heading"}

    def test_oversized_paragraph_falls_back_to_token_windows(self) -> None:
        chunker = SectionChunker(token_limit=2)
        document = _document(
            [_page(1, "Heading\nw0 w1 w2 w3 w4 w5", {"heading": "Heading"})]
        )

        chunks = chunker.chunk(document)

        assert [chunk.content for chunk in chunks] == [
            "Heading w0",
            "Heading\nw1 w2",
            "Heading\nw3 w4",
            "Heading\nw5",
        ]

    def test_chunk_index_is_continuous_across_merges_and_splits(self) -> None:
        chunker = SectionChunker(token_limit=2)
        document = _document(
            [
                _page(1, "a b", {"heading": "A"}),
                _page(2, "Heading\nw0 w1 w2 w3", {"heading": "Heading"}),
                _page(3, "c d", {"heading": "C"}),
            ]
        )

        chunks = chunker.chunk(document)

        assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2, 3, 4]
        assert [chunk.page_number for chunk in chunks] == [1, 2, 2, 2, 3]

    def test_empty_pages_are_skipped(self) -> None:
        chunker = SectionChunker(token_limit=100)
        document = _document([_page(1, ""), _page(2, "a b", {"heading": "B"})])

        chunks = chunker.chunk(document)

        assert len(chunks) == 1
        assert chunks[0].content == "a b"
        assert chunks[0].page_number == 2

    def test_document_without_pages_returns_no_chunks(self) -> None:
        chunker = SectionChunker(token_limit=100)

        assert chunker.chunk(_document([])) == []

    @pytest.mark.parametrize("token_limit", [0, -5])
    def test_invalid_config_raises(self, token_limit: int) -> None:
        with pytest.raises(ValueError):
            SectionChunker(token_limit=token_limit)


def test_markdown_document_pipeline_token_chunker(sample_markdown: Path) -> None:
    document = MarkdownParser().parse(sample_markdown)
    chunks = TokenChunker(token_limit=512, token_overlap=64).chunk(document)

    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]
    assert [chunk.page_number for chunk in chunks] == [1, 2, 3]
    assert [chunk.metadata["heading"] for chunk in chunks] == [
        "Iris KNN Classification",
        "Dataset",
        "Implementation",
    ]
    assert all(chunk.document_id == document.id for chunk in chunks)


def test_markdown_document_pipeline_section_chunker(sample_markdown: Path) -> None:
    document = MarkdownParser().parse(sample_markdown)
    chunks = SectionChunker(token_limit=512).chunk(document)

    assert len(chunks) == 1
    assert chunks[0].page_number == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].metadata["heading"] == "Iris KNN Classification"
    assert chunks[0].document_id == document.id
    assert "Dataset" in chunks[0].content
    assert "Implementation" in chunks[0].content
