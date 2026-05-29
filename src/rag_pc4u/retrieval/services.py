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

    results = pipeline.run({
        "dense_embedder": {"text": query},
        "sparse_embedder": {"text": query},
        "hybrid_retriever": {"filters": runtime_filters},
        "prompt_builder": {"query": query},
    })

    replies = results.get("llm", {}).get("replies", [])
    reply_text = replies[0] if replies else "Désolé, je n'ai pas pu générer de réponse."

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
        answer=reply_text,
        sources=sources,
        metadata={
            "client_id": client_id,
            "llm_model": settings.ollama_llm_model,
            "docs_retrieved": len(sources),
        },
    )