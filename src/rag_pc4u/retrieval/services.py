import structlog
from haystack import Pipeline
from rag_pc4u.core.models import QueryResponse, Source
from rag_pc4u.core.tenancy import filter_for
from rag_pc4u.core.config import settings
from haystack.dataclasses import ChatMessage

logger = structlog.get_logger(__name__)


def answer(query: str, client_id: str, pipeline: Pipeline) -> QueryResponse:
    """Isole l'API du format interne de Haystack en utilisant le pipeline injecté."""
    logger.info("Exécution du RAG", client_id=client_id, query=query)

    runtime_filters = filter_for(client_id=client_id)

    # C'est ici qu'on distribue manuellement la query au ranker et au prompt_builder
    results = pipeline.run({
        "dense_embedder": {"text": query},
        "sparse_embedder": {"text": query},
        "hybrid_retriever": {"filters": runtime_filters},
        "prompt_builder": {"query": query},
        "ranker": {"query": query},
    })

    replies = results.get("llm", {}).get("replies", [])

    if replies:
        if isinstance(replies[0], ChatMessage):
            reply_text = replies[0].content
        elif isinstance(replies[0], str):
            reply_text = replies[0]
        else:
            reply_text = str(replies[0])
    else:
        reply_text = "Désolé, je n'ai pas pu générer de réponse."

    retrieved_docs = results.get("document_exposer", {}).get("exposed_documents", [])

    sources = [
        Source(
            url=doc.meta.get("url", "Non spécifiée"),
            content=doc.content if doc.content else "",
            metadata=doc.meta,
        )
        for doc in retrieved_docs
    ]

    # DEBUG CRITIQUE
    print(f"DEBUG: Nombre de documents trouvés = {len(retrieved_docs)}")
    for i, doc in enumerate(retrieved_docs):
        print(f"DEBUG: Contenu du doc {i} : {doc.content[:100]}...")  # Affiche les 100 premiers caractères

    if len(retrieved_docs) > 0 and "Désolé" in reply_text:
        print("ALERTE: Le LLM a ignoré les documents fournis !")

    return QueryResponse(
        answer=reply_text,
        sources=sources,
        metadata={
            "client_id": client_id,
            "llm_model": settings.ollama_llm_model,
            "docs_retrieved": len(sources),
        },
    )