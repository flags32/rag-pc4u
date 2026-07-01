"""Couche service RAG — isole l'API du format interne Haystack."""
import structlog
from typing import AsyncIterator
from haystack.dataclasses import ChatMessage, StreamingChunk

from rag_pc4u.core.models import QueryResponse, Source
from rag_pc4u.core.config import settings
from rag_pc4u.retrieval.pipeline import build_hybrid_rag_pipeline

logger = structlog.get_logger(__name__)


def _run_retrieval(query: str, collection_name: str):
    """
    Exécute la partie retrieval + reranking du pipeline (sans le LLM).
    Réutilisé par answer() et answer_stream().
    """
    # with_llm=False : le composant "llm" n'existe même pas dans ce pipeline,
    # donc il ne peut pas être exécuté. Avant ce correctif, le pipeline complet
    # était utilisé et "include_outputs_from" ne faisait que filtrer la sortie
    # renvoyée -> le LLM tournait quand même une première fois pour rien
    # (double génération à chaque requête en streaming).
    pipeline = build_hybrid_rag_pipeline(collection_name, with_llm=False)

    results = pipeline.run(
        {
            "dense_embedder": {"text": query},
            "sparse_embedder": {"text": query},
            "prompt_builder": {"query": query},
            "ranker": {"query": query},
        },
        include_outputs_from={"prompt_builder", "document_exposer"},
    )

    prompt = results["prompt_builder"]["prompt"]
    retrieved_docs = results.get("document_exposer", {}).get("exposed_documents", [])

    sources = [
        Source(
            url=doc.meta.get("file_path", "Non spécifiée"),
            content=doc.content or "",
            metadata=doc.meta,
        )
        for doc in retrieved_docs
    ]

    return prompt, sources


def answer(query: str, collection_name: str) -> QueryResponse:
    """Version non-streaming (inchangée dans son comportement)."""
    logger.info("Exécution du RAG (non-stream)", collection=collection_name, query=query)

    pipeline = build_hybrid_rag_pipeline(collection_name)

    results = pipeline.run({
        "dense_embedder": {"text": query},
        "sparse_embedder": {"text": query},
        "prompt_builder": {"query": query},
        "ranker": {"query": query},
    })

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

    retrieved_docs = results.get("document_exposer", {}).get("exposed_documents", [])

    sources = [
        Source(
            url=doc.meta.get("file_path", "Non spécifiée"),
            content=doc.content or "",
            metadata=doc.meta,
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


async def answer_stream(query: str, collection_name: str) -> AsyncIterator[str]:
    """
    Génère la réponse RAG token par token (générateur async).

    Étapes :
    1. Retrieval + reranking + construction du prompt (synchrone, identique à answer()).
    2. Appel direct à OllamaGenerator avec streaming_callback pour récupérer
       chaque chunk dès qu'il arrive d'Ollama.

    Yields :
        str : chaque morceau de texte généré par le LLM, dans l'ordre.
    """
    import asyncio
    from haystack_integrations.components.generators.ollama import OllamaGenerator

    logger.info("Exécution du RAG (stream)", collection=collection_name, query=query)

    # 1. Retrieval (réutilise la même logique que answer())
    prompt, sources = _run_retrieval(query, collection_name)

    # 2. Queue pour faire le pont entre le callback synchrone d'Ollama
    #    et le générateur async consommé par FastAPI.
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def streaming_callback(chunk: StreamingChunk) -> None:
        # Ce callback est appelé de manière synchrone par OllamaGenerator,
        # potentiellement depuis un thread différent.
        asyncio.run_coroutine_threadsafe(queue.put(chunk.content), loop)

    generator = OllamaGenerator(
        url=settings.ollama_host,
        model=settings.ollama_llm_model,
        system_prompt=settings.__dict__.get("rag_system_prompt", None) or _system_prompt(),
        streaming_callback=streaming_callback,
    )

    async def _run_generator():
        try:
            # OllamaGenerator.run est bloquant -> on l'exécute dans un thread
            await loop.run_in_executor(None, generator.run, prompt)
        finally:
            await queue.put(None)  # Signal de fin

    task = asyncio.ensure_future(_run_generator())

    try:
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk
    finally:
        await task


def _system_prompt() -> str:
    from rag_pc4u.retrieval.prompts import RAG_SYSTEM_PROMPT
    return RAG_SYSTEM_PROMPT