"""Module de routes pour la santé de l'API RAG PC4U."""
import time
from typing import Any

import httpx
import structlog
from fastapi import APIRouter

from rag_pc4u.core.components import get_document_store
from rag_pc4u.core.config import settings

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["Health"])


async def _ping_qdrant() -> dict[str, Any]:
    """
    Vérifie que Qdrant répond et que la collection existe.
    Retourne un dict avec status et détail.
    """
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{settings.qdrant_url}/healthz")
        if resp.status_code == 200:
            ds = get_document_store()
            count = ds.count_documents()
            return {"status": "ok", "documents": count}
        return {"status": "error", "detail": f"HTTP {resp.status_code}"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


async def _ping_ollama() -> dict[str, Any]:
    """Vérifie qu'Ollama répond sur son endpoint racine."""
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(settings.ollama_host)
        if resp.status_code == 200:
            return {"status": "ok", "model": settings.ollama_llm_model}
        return {"status": "error", "detail": f"HTTP {resp.status_code}"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@router.get("/health")
async def health() -> dict[str, Any]:
    """
    Health check complet.

    Avant : retournait toujours 'ok' sans rien verifier.
    Maintenant : ping reel sur Qdrant et Ollama.
    Le statut global passe a 'degraded' si l'un des deux est KO.
    """
    qdrant_status = await _ping_qdrant()
    ollama_status = await _ping_ollama()

    all_ok = qdrant_status["status"] == "ok" and ollama_status["status"] == "ok"

    return {
        "status": "ok" if all_ok else "degraded",
        "timestamp": int(time.time()),
        "client_id": settings.client_id,
        "collection": settings.collection_name,
        "dependencies": {
            "qdrant": qdrant_status,
            "ollama": ollama_status,
        },
    }
