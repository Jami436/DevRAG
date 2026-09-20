from functools import lru_cache
from uuid import UUID

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.core.config import settings
from app.domain.retrieval.entities import RetrievedChunk
from app.services.search import SearchService, build_default_search_service

router = APIRouter()


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4096)
    top_k: int = Field(default=settings.retriever_top_k, ge=1, le=20)


class SearchResultResponse(BaseModel):
    chunk_id: UUID
    document_id: UUID
    document_title: str | None = None
    content: str
    chunk_index: int
    page_number: int
    metadata: dict[str, str] = Field(default_factory=dict)
    vector_score: float | None = None
    keyword_score: float | None = None
    final_score: float | None = None
    rank: int | None = None


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResultResponse] = Field(default_factory=list)


@lru_cache
def get_search_service() -> SearchService:
    """Build the default search service once per process."""
    return build_default_search_service(settings)


def _to_result(hit: RetrievedChunk) -> SearchResultResponse:
    return SearchResultResponse(
        chunk_id=hit.chunk_id,
        document_id=hit.document_id,
        document_title=hit.document_title,
        content=hit.content,
        chunk_index=hit.chunk_index,
        page_number=hit.page_number,
        metadata=hit.metadata,
        vector_score=hit.vector_score,
        keyword_score=hit.keyword_score,
        final_score=hit.final_score,
        rank=hit.rank,
    )


@router.post("/search", tags=["Search"])
def search(
    request: SearchRequest,
    search_service: SearchService = Depends(get_search_service),
) -> SearchResponse:
    """Retrieve and rerank the chunks most relevant to a question.

    The response exposes each hit with its per-signal scores (vector and
    keyword), the fused/reranked score and its final 1-based rank, so callers
    can inspect exactly why a chunk was surfaced.
    """
    results = search_service.execute(request.query, top_k=request.top_k)
    return SearchResponse(
        query=request.query,
        results=[_to_result(hit) for hit in results],
    )