"""Pipeline module Rag PC4U pour retrieval"""
from typing import List
from haystack import Pipeline, component
from haystack.components.builders import PromptBuilder
from haystack.dataclasses import Document

from haystack_integrations.components.embedders.fastembed import FastembedSparseTextEmbedder
from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder
from haystack_integrations.components.generators.ollama import OllamaGenerator
from haystack_integrations.components.retrievers.qdrant import QdrantHybridRetriever

from rag_pc4u.core.components import get_document_store
from rag_pc4u.core.config import settings
from rag_pc4u.retrieval.prompts import RAG_SYSTEM_PROMPT, RAG_USER_TEMPLATE
from rag_pc4u.retrieval.reranker import SimpleBGEReranker


@component
class DocumentExposer:
    """Composant utilitaire pour exposer les documents récupérés dans les résultats finaux."""
    @component.output_types(documents=List[Document], exposed_documents=List[Document])
    def run(self, documents: List[Document]):
        return {"documents": documents, "exposed_documents": documents}


def build_hybrid_rag_pipeline() -> Pipeline:
    """Assemble l'arborescence des composants Haystack pour la recherche hybride."""
    ds = get_document_store()
    pipeline = Pipeline()

    pipeline.add_component("dense_embedder", OllamaTextEmbedder(
        model=settings.ollama_embed_model,
        url=settings.ollama_host
    ))

    pipeline.add_component("sparse_embedder", FastembedSparseTextEmbedder(
        model="Qdrant/bm25", parallel=None
    ))

    pipeline.add_component("hybrid_retriever", QdrantHybridRetriever(
        document_store=ds, top_k=settings.top_k
    ))

    pipeline.add_component("document_exposer", DocumentExposer())

    # Retour au constructeur de Prompt classique
    pipeline.add_component(
        "prompt_builder",
        PromptBuilder(
            template=RAG_USER_TEMPLATE,
            required_variables=["documents", "query"]
        )
    )

    # Retour au générateur classique
    pipeline.add_component("llm", OllamaGenerator(
        url=settings.ollama_host,
        model=settings.ollama_llm_model,
        system_prompt=RAG_SYSTEM_PROMPT,
    ))

    pipeline.add_component("ranker", SimpleBGEReranker(top_k=2))

    """connexion des composants"""

    # 1. Connexions Embeddings
    pipeline.connect("dense_embedder.embedding", "hybrid_retriever.query_embedding")
    pipeline.connect("sparse_embedder.sparse_embedding", "hybrid_retriever.query_sparse_embedding")

    # 2. Retriever -> Ranker (On envoie les résultats bruts au reranker)
    pipeline.connect("hybrid_retriever.documents", "ranker.documents")

    # 3. Ranker -> Document Exposer (Pour l'API)
    pipeline.connect("ranker.documents", "document_exposer.documents")

    # 4. Ranker -> Prompt Builder (Pour que le LLM reçoive les docs TRIÉS)
    pipeline.connect("ranker.documents", "prompt_builder.documents")

    # 5. Connexion du prompt généré vers le LLM
    pipeline.connect("prompt_builder.prompt", "llm.prompt")

    return pipeline