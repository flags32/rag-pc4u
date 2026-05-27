"""Module de routes pour les requêtes de l'API RAG PC4U."""

from fastapi import APIRouter, Request, Depends, HTTPException
import structlog

from rag_pc4u.core.models import QueryRequest, QueryResponse
from rag_pc4u.retrieval.services import answer
from rag_pc4u.api.security import get_client_id_from_key

logger = structlog.get_logger(__name__)
router = APIRouter()

@router.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query_rag(
    request: Request,
    payload: QueryRequest,
    client_id: str = Depends(get_client_id_from_key)
):
    """
    Point d'entrée principal pour poser une question au RAG.
    Le client_id est déduit automatiquement de la clé d'API.
    """
    pipeline = request.app.state.query_pipeline
    if not pipeline:
        raise HTTPException(
            status_code=503,
            detail="Le pipeline de recherche n'est pas encore initialisé."
        )

    try:
        # Appel du service qui isole la logique Haystack
        response = answer(
            query=payload.question,
            client_id=client_id,
            pipeline=pipeline
        )
        return response

    except Exception as e:
        logger.error("api.query.error", error=str(e), client_id=client_id)
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la génération de la réponse: {str(e)}"
        )