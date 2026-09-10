from pathlib import Path

from app.domain.documents.entities import Document
from app.infrastructure.ingestion.pdf_parser import PDFParser


PDF_PATH = Path(
    r"F:\DevRAG-test-data\Iris KNN Classification Pipeline.pdf"
)


def test_pdf_parser_preserves_document_structure() -> None:
    parser = PDFParser()

    document = parser.parse(PDF_PATH)

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


def test_pdf_parser_preserves_page_content() -> None:
    parser = PDFParser()

    document = parser.parse(PDF_PATH)

    first_page = document.pages[0]
    fourth_page = document.pages[3]

    assert "Iris KNN Classification Pipeline" in first_page.content
    assert "The implementation separates responsibilities" in fourth_page.content


def test_pdf_parser_sets_metadata() -> None:
    parser = PDFParser()

    document = parser.parse(PDF_PATH)

    assert document.metadata["file_name"] == (
        "Iris KNN Classification Pipeline.pdf"
    )
    assert document.metadata["page_count"] == "6"