"""Pipeline RAG hybride PC4U — un pipeline par collection, mis en cache."""
from functools import lru_cache
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
    """
    Expose les documents reranked vers deux sorties :
    - documents        → prompt_builder (pour le LLM)
    - exposed_documents → services.py  (pour construire les sources retournées à l'API)
    """

    @component.output_types(documents=List[Document], exposed_documents=List[Document])
    def run(self, documents: List[Document]):
        return {"documents": documents, "exposed_documents": documents}


@lru_cache(maxsize=16)
def build_hybrid_rag_pipeline(collection_name: str, with_llm: bool = True) -> Pipeline:
    """
    Assemble le pipeline RAG hybride pour une collection donnée.

    Le décorateur @lru_cache garantit qu'une seule instance de pipeline est
    créée par (collection_name, with_llm), puis réutilisée pour toutes les requêtes.
    C'est services.py qui appelle cette fonction — pas la route HTTP.

    Args:
        collection_name : Nom de la collection Qdrant à interroger.
        with_llm : Si False, le composant "llm" n'est ni créé ni connecté.
                   Utilisé pour le chemin streaming (answer_stream), où le LLM
                   est appelé séparément avec son propre streaming_callback.
                   Sans ça, Haystack exécute quand même "llm" dès que
                   prompt_builder.prompt est disponible, même si on ne demande
                   pas sa sortie -> double génération inutile (~2x la latence).
    """
    ds = get_document_store(collection_name)
    pipeline = Pipeline()

    pipeline.add_component("dense_embedder", OllamaTextEmbedder(
        model=settings.ollama_embed_model,
        url=settings.ollama_host,
    ))
    pipeline.add_component("sparse_embedder", FastembedSparseTextEmbedder(
        model="Qdrant/bm25", parallel=None,
    ))
    pipeline.add_component("hybrid_retriever", QdrantHybridRetriever(
        document_store=ds,
        top_k=settings.top_k,
    ))
    pipeline.add_component("ranker", SimpleBGEReranker(top_k=10))
    pipeline.add_component("document_exposer", DocumentExposer())
    pipeline.add_component(
        "prompt_builder",
        PromptBuilder(
            template=RAG_USER_TEMPLATE,
            required_variables=["documents", "query"],
        ),
    )
    if with_llm:
        pipeline.add_component("llm", OllamaGenerator(
            url=settings.ollama_host,
            model=settings.ollama_llm_model,
            system_prompt=RAG_SYSTEM_PROMPT,
        ))

    # Connexions
    pipeline.connect("dense_embedder.embedding", "hybrid_retriever.query_embedding")
    pipeline.connect("sparse_embedder.sparse_embedding", "hybrid_retriever.query_sparse_embedding")
    pipeline.connect("hybrid_retriever.documents", "ranker.documents")
    pipeline.connect("ranker.documents", "document_exposer.documents")
    pipeline.connect("ranker.documents", "prompt_builder.documents")
    if with_llm:
        pipeline.connect("prompt_builder.prompt", "llm.prompt")

    return pipeline