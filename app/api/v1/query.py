from functools import lru_cache
from uuid import UUID

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.core.config import settings
from app.domain.generation.entities import Citation
from app.services.query import QueryService, build_default_query_service

router = APIRouter()


class CitationResponse(BaseModel):
    chunk_id: UUID
    document_id: UUID
    chunk_index: int
    page_number: int
    document_title: str | None = None
    excerpt: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4096)
    top_k: int = Field(default=settings.retriever_top_k, ge=1, le=20)


class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: list[CitationResponse] = Field(default_factory=list)
    model: str | None = None


@lru_cache
def get_query_service() -> QueryService:
    """Build the default query service once per process."""
    return build_default_query_service(settings)


def _to_response(citation: Citation) -> CitationResponse:
    return CitationResponse(
        chunk_id=citation.chunk_id,
        document_id=citation.document_id,
        chunk_index=citation.chunk_index,
        page_number=citation.page_number,
        document_title=citation.document_title,
        excerpt=citation.excerpt,
        metadata=citation.metadata,
    )


@router.post("/query", tags=["Query"])
async def ask_question(
    request: QueryRequest,
    query_service: QueryService = Depends(get_query_service),
) -> QueryResponse:
    """Answer a question from the indexed documentation with source citations.

    The request is answered by retrieving the most relevant chunks, reranking
    them, composing a grounded generation prompt and mapping the model's inline
    ``[n]`` references back onto the retrieved sources.
    """
    answer = query_service.execute(request.query, top_k=request.top_k)
    return QueryResponse(
        query=answer.query,
        answer=answer.answer,
        citations=[_to_response(citation) for citation in answer.citations],
        model=answer.model,
    )