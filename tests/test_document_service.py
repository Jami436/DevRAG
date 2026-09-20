from pathlib import Path
from uuid import UUID, uuid4

from app.domain.documents.entities import Document, DocumentPage
from app.infrastructure.db.models import ChunkModel, DocumentModel
from app.services.documents import DocumentService


class _FakePipeline:
    def __init__(self, document: Document) -> None:
        self.document = document
        self.calls: list[Path] = []

    def execute(self, file_path: Path) -> Document:
        self.calls.append(file_path)
        return self.document


class _FakeManager:
    def __init__(self) -> None:
        self.closed = 0
        self.commits = 0
        self.get_results: dict[UUID, DocumentModel | None] = {}
        self.delete_results: dict[UUID, bool] = {}
        self.chunks: list[ChunkModel] = []
        self.documents: list[DocumentModel] = []
        self.total = 0

    def get_document(self, document_id: UUID) -> DocumentModel | None:
        return self.get_results.get(document_id)

    def get_chunks(self, document_id: UUID) -> list[ChunkModel]:
        del document_id
        return self.chunks

    def list_documents(self, limit: int, offset: int) -> list[DocumentModel]:
        return self.documents

    def count_documents(self) -> int:
        return self.total

    def delete_document(self, document_id: UUID) -> bool:
        return self.delete_results.get(document_id, False)

    def commit(self) -> None:
        self.commits += 1

    def close(self) -> None:
        self.closed += 1


def _document() -> Document:
    return Document(
        title="Guide",
        source="guide.md",
        pages=[DocumentPage(page_number=1, content="content")],
    )


def _model() -> DocumentModel:
    return DocumentModel(
        id=uuid4(),
        title="Guide",
        source="guide.md",
        metadata_={"heading": "Intro"},
    )


def _service(
    pipeline: _FakePipeline, manager: _FakeManager
) -> DocumentService:
    return DocumentService(pipeline, lambda: manager)


def test_ingest_forwards_file_path_to_pipeline() -> None:
    pipeline = _FakePipeline(_document())
    manager = _FakeManager()
    service = _service(pipeline, manager)
    file_path = Path("docs", "guide.md")

    document = service.ingest(file_path)

    assert document is pipeline.document
    assert pipeline.calls == [file_path]


def test_get_returns_document_model_and_closes_repository() -> None:
    model = _model()
    manager = _FakeManager()
    manager.get_results[model.id] = model
    service = _service(_FakePipeline(_document()), manager)

    assert service.get(model.id) is model
    assert manager.closed == 1


def test_get_returns_none_for_missing_document() -> None:
    manager = _FakeManager()
    manager.get_results[uuid4()] = None
    service = _service(_FakePipeline(_document()), manager)

    assert service.get(uuid4()) is None
    assert manager.closed == 1


def test_get_chunks_returns_stored_chunks() -> None:
    chunk = ChunkModel(
        id=uuid4(),
        document_id=uuid4(),
        content="content",
        chunk_index=0,
        page_number=1,
    )
    manager = _FakeManager()
    manager.chunks = [chunk]
    service = _service(_FakePipeline(_document()), manager)

    assert service.get_chunks(chunk.document_id) == [chunk]
    assert manager.closed == 1


def test_list_returns_items_and_total() -> None:
    manager = _FakeManager()
    manager.documents = [_model(), _model()]
    manager.total = 2
    service = _service(_FakePipeline(_document()), manager)

    items, total = service.list(limit=10, offset=0)

    assert len(items) == 2
    assert total == 2
    assert manager.closed == 1


def test_delete_returns_whether_document_existed() -> None:
    existing = uuid4()
    missing = uuid4()
    manager = _FakeManager()
    manager.delete_results[existing] = True
    manager.delete_results[missing] = False
    service = _service(_FakePipeline(_document()), manager)

    assert service.delete(existing) is True
    assert manager.commits == 1
    assert service.delete(missing) is False
    assert manager.commits == 1
    assert manager.closed == 2