from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import structlog
import uvicorn
from fastapi import FastAPI

from rag_pc4u.core.config import settings
from rag_pc4u.core.logging import configure_logging

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    logger.info(
        "api.startup",
        client_id=settings.client_id,
        collection=settings.collection_name,
        qdrant=settings.qdrant_url,
        ollama=settings.ollama_host,
    )
    yield
    logger.info("api.shutdown")


app = FastAPI(
    title="RAG PC4U API",
    version="0.1.0",
    description="API RAG on-premise souveraine — stack Haystack + Qdrant + Ollama",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "client_id": settings.client_id,
        "collection": settings.collection_name,
    }


def run() -> None:
    uvicorn.run(
        "rag_pc4u.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )


if __name__ == "__main__":
    run()
