from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "DevRAG"
    app_version: str = "0.1.0"
    debug: bool = True
    api_v1_prefix: str = "/api/v1"

    database_url: str = "postgresql+psycopg://devrag:devrag@localhost:5432/devrag"

    chunking_strategy: str = "section"
    chunking_token_limit: int = 512
    chunking_token_overlap: int = 64

    embedding_provider: str = "openai"
    openai_api_key: str | None = None
    openai_embedding_model: str = "text-embedding-3-small"
    local_embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 1536
    embedding_batch_size: int = 2048

    retriever_top_k: int = 5
    retrieval_hybrid_candidates: int = 50
    retrieval_hybrid_vector_weight: float = 1.0
    retrieval_hybrid_keyword_weight: float = 1.0
    retrieval_fusion_k: int = 60

    reranker_provider: str = "cross_encoder"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    llm_reranker_model: str = "gpt-4o-mini"

    evaluation_golden_queries_path: str = "data/evaluation/golden_queries.json"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )


settings = Settings()
