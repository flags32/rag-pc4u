"""Services module Rag PC4U pour retrieval"""
# retrieval/service.py
import structlog
from haystack import Pipeline
from rag_pc4u.core.models import QueryResponse, Source
from rag_pc4u.core.tenancy import filter_for
from rag_pc4u.core.config import settings

logger = structlog.get_logger(__name__)


def answer(query: str, client_id: str, pipeline: Pipeline) -> QueryResponse:
    """Isole l'API du format interne de Haystack en utilisant le pipeline injecté."""
    logger.info("Exécution du RAG", client_id=client_id, query=query)

    runtime_filters = filter_for(client_id=client_id)

    # On exécute le pipeline PASSE EN PARAMÈTRE (déjà instancié)
    results = pipeline.run({
        "dense_embedder": {"text": query},
        "sparse_embedder": {"text": query},
        "hybrid_retriever": {"filters": runtime_filters},
        "prompt_builder": {"query": query},
    })

    # Parsing sécurisé de la réponse LLM
    replies = results.get("llm", {}).get("replies", [])
    answer = replies[0] if replies else "Désolé, je n'ai pas pu générer de réponse."

    # Conversion des documents Haystack vers ton modèle Pydantic 'Source'
    retrieved_docs = results.get("hybrid_retriever", {}).get("documents", [])
    sources = [
        Source(
            url=doc.meta.get("url", "Non spécifiée"),
            content=doc.content if doc.content else "",
            metadata=doc.meta,
        )
        for doc in retrieved_docs
    ]

    return QueryResponse(
        answer=answer,
        sources=sources,
        metadata={
            "client_id": client_id,
            "llm_model": settings.ollama_llm_model,
            "docs_retrieved": len(sources)
        },
    )