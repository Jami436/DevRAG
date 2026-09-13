from uuid import UUID

from app.domain.documents.chunks import DocumentChunk


def test_document_chunk_creation() -> None:
    document_id = UUID("12345678-1234-5678-1234-567812345678")

    chunk = DocumentChunk(
        document_id=document_id,
        content="KNN is a supervised learning algorithm.",
        chunk_index=0,
        page_number=3,
    )

    assert chunk.document_id == document_id
    assert chunk.content == "KNN is a supervised learning algorithm."
    assert chunk.chunk_index == 0
    assert chunk.page_number == 3
    assert isinstance(chunk.id, UUID)


def test_document_chunk_metadata_defaults_to_empty() -> None:
    document_id = UUID("12345678-1234-5678-1234-567812345678")

    chunk = DocumentChunk(
        document_id=document_id,
        content="Example chunk",
        chunk_index=1,
        page_number=2,
    )

    assert chunk.metadata == {}
