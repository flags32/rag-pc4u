"""Schémas de données pour l'API RAG PC4U."""
from pydantic import Field, BaseModel
from rag_pc4u.core.models import QueryRequest, QueryResponse


class HTTPQueryRequest(QueryRequest):
    """Validation de la requête entrante pour le RAG."""
    question: str = Field(
        ...,
        description="La question à poser au RAG",
        examples=["Comment configurer le VPN de mon poste ?"],
    )


class HTTPQueryResponse(QueryResponse):
    """Structure de la réponse renvoyée par le RAG."""
    pass


__all__ = ["HTTPQueryRequest", "HTTPQueryResponse"]



