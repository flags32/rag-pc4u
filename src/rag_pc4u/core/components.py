import os
import structlog
from typing import Any
from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack_integrations.components.embedders.fastembed import FastembedSparseTextEmbedder
from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder
from haystack_integrations.components.generators.ollama import OllamaGenerator
from haystack_integrations.components.retrievers.qdrant import QdrantHybridRetriever
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

from rag_pc4u.core.config import settings
from rag_pc4u.core.models import QueryResponse, Source
from rag_pc4u.core.tenancy import filter_for

logger = structlog.get_logger(__name__)

# Pattern Singleton propre pour le Document Store
_document_store: QdrantDocumentStore | None = None


def get_document_store() -> QdrantDocumentStore:
    """Initialise de manière unique et sécurisée le Document Store Qdrant."""
    global _document_store
    if _document_store is None:
        logger.info(
            "Connexion au QdrantDocumentStore",
            url=settings.qdrant_url,
            collection=settings.collection_name,
        )
        # Utilisation de l'initialisation recommandée pour Haystack 2.x
        _document_store = QdrantDocumentStore(
            url=settings.qdrant_url,
            index=settings.collection_name,
            embedding_dim=settings.embedding_dim,
            use_sparse_embeddings=True,  # INDISPENSABLE pour le mode Hybride
            recreate_index=False,  # NE PAS recréer à chaque import de composant
        )
    return _document_store


def build_hybrid_rag_pipeline() -> Pipeline:
    """Assemble l'arborescence des composants Haystack pour la recherche hybride."""
    ds = get_document_store()
    pipeline = Pipeline()

    # Branche Dense (Sémantique via Ollama)
    pipeline.add_component(
        "dense_embedder",
        OllamaTextEmbedder(url=settings.ollama_host, model=settings.ollama_embed_model),
    )

    # Branche Sparse (Mots-clés via Fastembed)
    pipeline.add_component(
        "sparse_embedder",
        FastembedSparseTextEmbedder(model_name="BAAI/bge-m3")
    )

    # Retriever Hybride unifié
    pipeline.add_component(
        "hybrid_retriever",
        QdrantHybridRetriever(document_store=ds, top_k=settings.top_k),
    )

    # Génération
    template = """
    Tu es un assistant virtuel intelligent. Réponds à la question en te basant uniquement sur le contexte fourni.
    Si tu ne trouves pas la réponse dans le contexte, dis poliment que tu ne sais pas.

    Contexte :
    {% for doc in documents %}
      {{ doc.content }}
    {% endfor %}

    Question : {{ query }}
    Réponse :
    """
    pipeline.add_component("prompt_builder", PromptBuilder(template=template))
    pipeline.add_component(
        "llm",
        OllamaGenerator(url=settings.ollama_host, model=settings.ollama_llm_model),
    )

    # Connexions du graphe
    pipeline.connect("dense_embedder.embedding", "hybrid_retriever.query_embedding")
    pipeline.connect("sparse_embedder.sparse_embedding", "hybrid_retriever.query_sparse_embedding")
    pipeline.connect("hybrid_retriever.documents", "prompt_builder.documents")
    pipeline.connect("prompt_builder.prompt", "llm.prompt")

    return pipeline


def execute_query(question: str, client_id: str) -> QueryResponse:
    """Point d'entrée d'exécution du RAG par rapport à un contexte client filtré."""
    logger.info("Traitement de la requête utilisateur", client_id=client_id, query=question)

    # Génération du dictionnaire de filtres conforme à tenancy.py
    runtime_filters = filter_for(client_id=client_id)

    pipeline = build_hybrid_rag_pipeline()

    # Exécution synchrone du pipeline
    results = pipeline.run(
        {
            "dense_embedder": {"text": question},
            "sparse_embedder": {"text": question},
            "hybrid_retriever": {"filters": runtime_filters},
            "prompt_builder": {"query": question},
        }
    )

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