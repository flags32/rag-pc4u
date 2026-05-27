"""Module de sécurité pour l'API RAG PC4U. avec Gestion de la sécurité et de l'authentification de l'API."""

from fastapi import Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader
from rag_pc4u.core.config import settings

# On définit le nom du header attendu pour la clé d'API
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Dans un vrai scénario, ces clés seraient dans une base de données ou un coffre-fort
# Pour l'instant, on simule un dictionnaire statique
VALID_API_KEYS = {
    "sk-demo-12345": "client_demo",
    "sk-pc4u-98765": "client_premium"
}


async def get_client_id_from_key(api_key: str = Security(api_key_header)) -> str:
    """
    Vérifie la clé d'API et retourne le client_id associé.
    À injecter dans les routes qui nécessitent une authentification.
    """
    if not api_key:
        # En mode dev/demo, on autorise le passage sans clé
        return settings.client_id

    client_id = VALID_API_KEYS.get(api_key)
    if not client_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Clé d'API invalide ou révoquée."
        )
    return client_id