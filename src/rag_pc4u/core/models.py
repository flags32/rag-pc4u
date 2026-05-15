from typing import Any

from pydantic import BaseModel, Field


class Source(BaseModel):
    """model pour représenter les sources de données extraite des documents haystack."""
    url: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    """model pour représenter les réponses aux requêtes avec les résultats et les méta-données."""
    answer: str
    sources: list[Source]
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    """model pour représenter les requêtes avec le texte de la question et les méta-données."""
    question: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """model pour représenter les réponses d'erreur avec un message détaillé."""
    detail: str
