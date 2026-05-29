from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import structlog
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from rag_pc4u.api.routes.query import router as query_router
from rag_pc4u.api.routes.ingest import router as ingest_router
from rag_pc4u.api.routes.health import router as health_router
from rag_pc4u.retrieval.pipeline import build_hybrid_rag_pipeline
from rag_pc4u.core.config import settings
from rag_pc4u.core.logger_config import configure_logging


logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    application.state.query_pipeline = build_hybrid_rag_pipeline()
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # A restreindre en production a l'URL de l'Open WebUI
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Montage des routes pour que le  main ne contient plus aucune logique metier
app.include_router(health_router)
app.include_router(query_router)
app.include_router(ingest_router)


def run() -> None:
    uvicorn.run(
        "rag_pc4u.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )


if __name__ == "__main__":
    run()
