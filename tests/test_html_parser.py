from pathlib import Path

import pytest

from app.domain.documents.entities import Document
from app.infrastructure.ingestion.html_parser import HTMLParser


def test_html_parser_preserves_document_structure(sample_html: Path) -> None:
    parser = HTMLParser()

    document = parser.parse(sample_html)

    assert isinstance(document, Document)
    assert document.title == "Iris KNN Classification"
    assert document.source == "html"
    assert document.page_count == 3


def test_html_parser_page_numbers_are_sequential(sample_html: Path) -> None:
    parser = HTMLParser()

    document = parser.parse(sample_html)

    assert [page.page_number for page in document.pages] == [1, 2, 3]


def test_html_parser_heading_prefix_appears_in_content(sample_html: Path) -> None:
    parser = HTMLParser()

    document = parser.parse(sample_html)

    first = document.pages[0]
    assert first.content.startswith("Iris KNN Classification\n")
    assert "end-to-end walkthrough" in first.content

    second = document.pages[1]
    assert second.content.startswith("Dataset\n")
    assert "dataset" in second.content


def test_html_parser_metadata_heading_matches_content_heading(
    sample_html: Path,
) -> None:
    parser = HTMLParser()

    document = parser.parse(sample_html)

    assert document.pages[0].metadata["heading"] == "Iris KNN Classification"
    assert document.pages[1].metadata["heading"] == "Dataset"
    assert document.pages[2].metadata["heading"] == "Implementation"


def test_html_parser_strips_inline_markup(sample_html: Path) -> None:
    parser = HTMLParser()

    document = parser.parse(sample_html)

    dataset_page = document.pages[1]
    assert "<strong>" not in dataset_page.content
    assert "KNN" in dataset_page.content


def test_html_parser_preserves_code_block(sample_html: Path) -> None:
    parser = HTMLParser()

    document = parser.parse(sample_html)

    impl_page = document.pages[2]
    assert "class KNN:" in impl_page.content
    assert "return self._nearest(x)" in impl_page.content


def test_html_parser_extracts_title_tag(sample_html: Path) -> None:
    parser = HTMLParser()

    document = parser.parse(sample_html)

    assert document.title == "Iris KNN Classification"


def test_html_parser_falls_back_to_stem_without_title(
    tmp_path: Path,
) -> None:
    path = tmp_path / "no_title.html"
    path.write_text("<html><body><h1>Heading</h1><p>Text</p></body></html>")

    document = HTMLParser().parse(path)

    assert document.title == "no_title"


def test_html_parser_sets_metadata(sample_html: Path) -> None:
    parser = HTMLParser()

    document = parser.parse(sample_html)

    assert document.metadata["file_name"] == "Iris KNN Classification.html"
    assert document.metadata["section_count"] == "3"


def test_html_parser_raises_on_missing_file(tmp_path: Path) -> None:
    parser = HTMLParser()

    with pytest.raises(FileNotFoundError):
        parser.parse(tmp_path / "missing.html")


def test_html_parser_rejects_non_html_files(tmp_path: Path) -> None:
    unknown_file = tmp_path / "notes.txt"
    unknown_file.write_text("hello", encoding="utf-8")

    parser = HTMLParser()

    with pytest.raises(ValueError):
        parser.parse(unknown_file)
