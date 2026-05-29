"""Pipeline module Rag PC4U pour retrieval"""
from haystack import Pipeline
from haystack.components.builders import PromptBuilder

from haystack_integrations.components.embedders.fastembed import FastembedSparseTextEmbedder
from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder

from haystack_integrations.components.generators.ollama import OllamaGenerator
from haystack_integrations.components.retrievers.qdrant import QdrantHybridRetriever

from rag_pc4u.core.components import get_document_store
from rag_pc4u.core.config import settings
from rag_pc4u.retrieval.prompts import RAG_SYSTEM_PROMPT


def build_hybrid_rag_pipeline() -> Pipeline:
    """Assemble l'arborescence des composants Haystack pour la recherche hybride."""
    ds = get_document_store()
    pipeline = Pipeline()

    # Branche Dense (Sémantique via Ollama)
    pipeline.add_component("dense_embedder", OllamaTextEmbedder(
        model=settings.ollama_embed_model,
        url=settings.ollama_host
    ))

    # Branche Sparse (Mots-clés via Fastembed)
    pipeline.add_component(
        "sparse_embedder",
        FastembedSparseTextEmbedder(model="Qdrant/bm25", parallel=1)
    )

    # Retriever Hybride unifié
    pipeline.add_component(
        "hybrid_retriever",
        QdrantHybridRetriever(document_store=ds, top_k=settings.top_k),
    )

    # Génération
    pipeline.add_component("prompt_builder", PromptBuilder(
        template=RAG_SYSTEM_PROMPT,
        required_variables=["documents", "query"],
    ))

    pipeline.add_component(
        "llm",
        OllamaGenerator(url=settings.ollama_host, model=settings.ollama_llm_model),
    )

    # Connexions du graphe (Ce câblage va maintenant fonctionner)
    pipeline.connect("dense_embedder.embedding", "hybrid_retriever.query_embedding")
    pipeline.connect("sparse_embedder.sparse_embedding", "hybrid_retriever.query_sparse_embedding")
    pipeline.connect("hybrid_retriever.documents", "prompt_builder.documents")
    pipeline.connect("prompt_builder.prompt", "llm.prompt")

    return pipeline