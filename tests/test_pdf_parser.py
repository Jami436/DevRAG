from pathlib import Path

import pytest

from app.domain.documents.entities import Document
from app.infrastructure.ingestion.pdf_parser import PDFParser


def test_pdf_parser_preserves_document_structure(sample_pdf: Path) -> None:
    parser = PDFParser()

    document = parser.parse(sample_pdf)

    assert isinstance(document, Document)
    assert document.title == "Iris KNN Classification Pipeline"
    assert document.source == "pdf"

    assert document.page_count == 6
    assert len(document.pages) == 6

    assert [page.page_number for page in document.pages] == [
        1,
        2,
        3,
        4,
        5,
        6,
    ]

    assert all(page.content for page in document.pages)


def test_pdf_parser_preserves_page_content(sample_pdf: Path) -> None:
    parser = PDFParser()

    document = parser.parse(sample_pdf)

    first_page = document.pages[0]
    fourth_page = document.pages[3]

    assert "Iris KNN Classification Pipeline" in first_page.content
    assert "The implementation separates responsibilities" in fourth_page.content


def test_pdf_parser_sets_metadata(sample_pdf: Path) -> None:
    parser = PDFParser()

    document = parser.parse(sample_pdf)

    assert document.metadata["file_name"] == ("Iris KNN Classification Pipeline.pdf")
    assert document.metadata["page_count"] == "6"


def test_pdf_parser_raises_on_missing_file(tmp_path: Path) -> None:
    parser = PDFParser()

    with pytest.raises(FileNotFoundError):
        parser.parse(tmp_path / "missing.pdf")


def test_pdf_parser_rejects_non_pdf_files(tmp_path: Path) -> None:
    unknown_file = tmp_path / "notes.txt"
    unknown_file.write_text("hello", encoding="utf-8")

    parser = PDFParser()

    with pytest.raises(ValueError):
        parser.parse(unknown_file)
