from datetime import UTC, datetime
from uuid import UUID

from app.domain.documents.entities import Document, DocumentPage


def test_document_page_fields() -> None:
    page = DocumentPage(page_number=3, content="Some text")

    assert page.page_number == 3
    assert page.content == "Some text"
    assert page.metadata == {}


def test_document_page_custom_metadata() -> None:
    page = DocumentPage(
        page_number=1,
        content="Hello",
        metadata={"source": "pdf"},
    )

    assert page.metadata == {"source": "pdf"}


def test_document_creation_and_page_count() -> None:
    pages = [
        DocumentPage(page_number=1, content="Intro"),
        DocumentPage(page_number=2, content="Body"),
    ]

    document = Document(title="Example", source="markdown", pages=pages)

    assert document.title == "Example"
    assert document.source == "markdown"
    assert document.pages == pages
    assert document.page_count == 2
    assert document.metadata == {}
    assert document.source_url is None
    assert isinstance(document.id, UUID)
    assert isinstance(document.created_at, datetime)


def test_document_empty_pages() -> None:
    document = Document(title="Empty", source="txt", pages=[])

    assert document.page_count == 0


def test_document_default_metadata_is_empty() -> None:
    document = Document(title="A", source="md", pages=[])

    assert document.metadata == {}


def test_document_default_source_url_is_none() -> None:
    document = Document(title="A", source="md", pages=[])

    assert document.source_url is None


def test_document_ids_are_unique() -> None:
    a = Document(title="A", source="md", pages=[])
    b = Document(title="B", source="md", pages=[])

    assert a.id != b.id


def test_document_created_at_is_utc() -> None:
    document = Document(title="A", source="md", pages=[])

    assert document.created_at.tzinfo is UTC
