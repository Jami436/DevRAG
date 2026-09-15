from pathlib import Path

import pytest

from app.domain.documents.entities import Document
from app.infrastructure.ingestion.markdown_parser import MarkdownParser


def test_markdown_parser_preserves_document_structure(
    sample_markdown: Path,
) -> None:
    parser = MarkdownParser()

    document = parser.parse(sample_markdown)

    assert isinstance(document, Document)
    assert document.title == "Iris KNN Classification"
    assert document.source == "markdown"
    assert document.page_count == 3


def test_markdown_parser_page_numbers_are_sequential(
    sample_markdown: Path,
) -> None:
    parser = MarkdownParser()

    document = parser.parse(sample_markdown)

    assert [page.page_number for page in document.pages] == [1, 2, 3]


def test_markdown_parser_heading_prefix_appears_in_content(
    sample_markdown: Path,
) -> None:
    parser = MarkdownParser()

    document = parser.parse(sample_markdown)

    first = document.pages[0]
    assert first.content.startswith("Iris KNN Classification\n")
    assert "end-to-end walkthrough" in first.content

    second = document.pages[1]
    assert second.content.startswith("Dataset\n")
    assert "dataset" in second.content


def test_markdown_parser_metadata_heading_matches_content_heading(
    sample_markdown: Path,
) -> None:
    parser = MarkdownParser()

    document = parser.parse(sample_markdown)

    assert document.pages[0].metadata["heading"] == "Iris KNN Classification"
    assert document.pages[1].metadata["heading"] == "Dataset"
    assert document.pages[2].metadata["heading"] == "Implementation"


def test_markdown_parser_strips_inline_markup(
    sample_markdown: Path,
) -> None:
    parser = MarkdownParser()

    document = parser.parse(sample_markdown)

    dataset_page = document.pages[1]
    assert "**" not in dataset_page.content
    assert "dataset" in dataset_page.content


def test_markdown_parser_preserves_code_fence(
    sample_markdown: Path,
) -> None:
    parser = MarkdownParser()

    document = parser.parse(sample_markdown)

    impl_page = document.pages[2]
    assert "class KNN:" in impl_page.content
    assert "return self._nearest(x)" in impl_page.content


def test_markdown_parser_sets_metadata(sample_markdown: Path) -> None:
    parser = MarkdownParser()

    document = parser.parse(sample_markdown)

    assert document.metadata["file_name"] == "Iris KNN Classification.md"
    assert document.metadata["section_count"] == "3"


def test_markdown_parser_raises_on_missing_file(tmp_path: Path) -> None:
    parser = MarkdownParser()

    with pytest.raises(FileNotFoundError):
        parser.parse(tmp_path / "missing.md")


def test_markdown_parser_rejects_non_markdown_files(tmp_path: Path) -> None:
    unknown_file = tmp_path / "notes.txt"
    unknown_file.write_text("hello", encoding="utf-8")

    parser = MarkdownParser()

    with pytest.raises(ValueError):
        parser.parse(unknown_file)


def test_markdown_parser_handles_file_without_headings(tmp_path: Path) -> None:
    path = tmp_path / "no_headings.md"
    path.write_text("Just some plain text.\nAnother line.", encoding="utf-8")

    document = MarkdownParser().parse(path)

    assert document.page_count == 1
    assert "Just some plain text." in document.pages[0].content
    assert document.pages[0].metadata["heading"] == ""
