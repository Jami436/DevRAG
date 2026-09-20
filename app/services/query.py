from app.application.generation.generate_answer import GenerateAnswer
from app.core.settings import Settings, settings
from app.domain.generation.entities import GeneratedAnswer
from app.domain.generation.interfaces import LLMProvider
from app.infrastructure.llm.factory import build_llm_provider
from app.services.search import SearchService, build_default_search_service


class QueryService:
    """End-to-end question answering over indexed documents.

    Delegates retrieval (embed, hybrid search, rerank) to a ``SearchService``
    and adds grounded answer generation with source citations on top. ``top_k``
    is forwarded to the search stage; the generation context budget is fixed at
    construction time.
    """

    def __init__(
        self,
        search_service: SearchService,
        llm_provider: LLMProvider,
        max_excerpt_chars: int = 1000,
        max_total_chars: int = 12000,
    ) -> None:
        self._search_service = search_service
        self._llm_provider = llm_provider
        self._max_excerpt_chars = max_excerpt_chars
        self._max_total_chars = max_total_chars

    def execute(self, query: str, *, top_k: int | None = None) -> GeneratedAnswer:
        chunks = self._search_service.execute(query, top_k=top_k)
        return GenerateAnswer(
            llm_provider=self._llm_provider,
            max_excerpt_chars=self._max_excerpt_chars,
            max_total_chars=self._max_total_chars,
        ).execute(query, chunks)


def build_default_query_service(app_settings: Settings = settings) -> QueryService:
    """Wire the default query service from application settings."""
    return QueryService(
        search_service=build_default_search_service(app_settings),
        llm_provider=build_llm_provider(app_settings),
        max_excerpt_chars=app_settings.generation_max_excerpt_chars,
        max_total_chars=app_settings.generation_max_context_chars,
    )