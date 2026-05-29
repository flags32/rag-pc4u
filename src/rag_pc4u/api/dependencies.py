"""Dépendances pour l'API RAG PC4U."""

from fastapi import Request, HTTPException, Header, status
from haystack import Pipeline

# Dépendance pour extraire le client_id
async def get_current_client_id(x_client_id: str = Header(None)) -> str:
    """Extrait et valide l'identifiant du client depuis les headers HTTP."""
    if not x_client_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Header 'X-Client-Id' manquant."
        )

    return x_client_id

# Dépendance pour récupérer le pipeline Haystack prêt à l'emploi
async def get_query_pipeline(request: Request) -> Pipeline:
    """
    Récupère le pipeline de requêtage Haystack stocké dans le state de l'application.
    Lève une erreur 503 si le pipeline n'est pas encore initialisé.
    """
    pipeline = getattr(request.app.state, "query_pipeline", None)

    if pipeline is None :
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Le pipeline de recherche n'est pas encore initialisé."
        )

    return pipeline