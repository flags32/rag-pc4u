"""Couche service RAG — isole l'API du format interne Haystack."""
import structlog
from haystack.dataclasses import ChatMessage

from rag_pc4u.core.models import QueryResponse, Source
from rag_pc4u.core.config import settings
from rag_pc4u.retrieval.pipeline import build_hybrid_rag_pipeline

logger = structlog.get_logger(__name__)


def answer(query: str, collection_name: str) -> QueryResponse:
    """
    Exécute le pipeline RAG sur la collection demandée et retourne une QueryResponse.

    Le pipeline est récupéré depuis le cache lru_cache de build_hybrid_rag_pipeline —
    pas de reconstruction à chaque appel.

    La route HTTP ne connaît ni les pipelines ni le format Haystack.
    Elle appelle answer(), reçoit une QueryResponse, c'est tout.

    Args:
        query           : Question de l'utilisateur.
        collection_name : Collection Qdrant à interroger.
    """
    logger.info("Exécution du RAG", collection=collection_name, query=query)

    pipeline = build_hybrid_rag_pipeline(collection_name)

    results = pipeline.run({
        "dense_embedder": {"text": query},
        "sparse_embedder": {"text": query},
        "prompt_builder": {"query": query},
        "ranker": {"query": query},
    })

    # Extraction de la réponse LLM (OllamaGenerator peut retourner str ou ChatMessage)
    replies = results.get("llm", {}).get("replies", [])
    if replies:
        first = replies[0]
        if isinstance(first, ChatMessage):
            reply_text = first.content
        elif isinstance(first, str):
            reply_text = first
        else:
            reply_text = str(first)
    else:
        reply_text = "Désolé, je n'ai pas pu générer de réponse."

    # Extraction des documents reranked exposés par DocumentExposer
    retrieved_docs = results.get("document_exposer", {}).get("exposed_documents", [])

    sources = [
        Source(
            # file_path comme URL de source (chemin complet pour traçabilité)
            url=doc.meta.get("file_path", "Non spécifiée"),
            content=doc.content or "",
            metadata=doc.meta,  # contient aussi file_name et date_added
        )
        for doc in retrieved_docs
    ]

    return QueryResponse(
        answer=reply_text,
        sources=sources,
        metadata={
            "collection": collection_name,
            "llm_model": settings.ollama_llm_model,
            "docs_retrieved": len(sources),
        },
    )