from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import structlog
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import QdrantClient
import yaml
from pathlib import Path
from rag_pc4u.api.routes.query import router as query_router
from rag_pc4u.api.routes.ingest import router as ingest_router
from rag_pc4u.api.routes.health import router as health_router
from rag_pc4u.core.config import settings
from rag_pc4u.core.logger_config import configure_logging

logger = structlog.get_logger(__name__)
"""
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    spec_path = Path(__file__).resolve().parents[2] / "docs" / "openapi_rag_pc4u.yaml"
    with open(spec_path) as f:
        app.openapi_schema = yaml.safe_load(f)
    return app.openapi_schema

app.openapi = custom_openapi 
"""
@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    configure_logging()

    # --- NOUVEAU : Récupération des collections existantes ---
    try:
        client = QdrantClient(url=settings.qdrant_url)
        response = client.get_collections()
        # On met à jour la liste de collection qui ce trouve dans les settings et qui redemarre vide a cahque lancement
        settings.List_collection = [col.name for col in response.collections]
        logger.info(
            "Collections synchronisées depuis Qdrant",
            count=len(settings.List_collection),
            collections=settings.List_collection
        )
    except Exception as e:
        logger.error("Erreur lors de la récupération des collections Qdrant", error=str(e))


    logger.info(
        "api.startup",
        qdrant=settings.qdrant_url,
        ollama=settings.ollama_host,
    )
    yield
    logger.info("api.shutdown")

app = FastAPI(
    title="RAG PC4U API v2",
    version="0.1.0",
    description="API RAG on-premise souveraine — multi-collections",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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