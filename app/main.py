from fastapi import FastAPI

from app.api.v1.documents import router as documents_router
from app.api.v1.health import router as health_router
from app.api.v1.query import router as query_router
from app.api.v1.search import router as search_router
from app.core.config import settings
from app.core.logging import setup_logging
from app.core.middleware import request_logging_middleware

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
)

setup_logging(settings)
app.middleware("http")(request_logging_middleware)

app.include_router(health_router, prefix=settings.api_v1_prefix)
app.include_router(documents_router, prefix=settings.api_v1_prefix)
app.include_router(search_router, prefix=settings.api_v1_prefix)
app.include_router(query_router, prefix=settings.api_v1_prefix)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_config=None,
        reload=settings.debug,
    )