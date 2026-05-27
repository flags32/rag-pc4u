"""Module de routes pour la santé de l'API RAG PC4U."""

import httpx
from fastapi import APIRouter
from rag_pc4u.core.config import settings

router = APIRouter()

@router.get("/health", tags=["System"])
async def health_check():
    """Vérifie l'état de l'API et de ses dépendances (Qdrant, Ollama)."""
    status_qdrant = "inconnu"
    status_ollama = "inconnu"

    async with httpx.AsyncClient(timeout=2.0) as client:
        # Check Qdrant
        try:
            r_qdrant = await client.get(f"{settings.qdrant_url}/readyz")
            status_qdrant = "up" if r_qdrant.status_code == 200 else "down"
        except Exception:
            status_qdrant = "unreachable"

        # Check Ollama
        try:
            r_ollama = await client.get(settings.ollama_host)
            status_ollama = "up" if r_ollama.status_code == 200 else "down"
        except Exception:
            status_ollama = "unreachable"

    overall_status = "ok" if (status_qdrant == "up" and status_ollama == "up") else "degraded"

    return {
        "status": overall_status,
        "client_id": settings.client_id,
        "collection": settings.collection_name,
        "dependencies": {
            "qdrant": status_qdrant,
            "ollama": status_ollama
        }
    }