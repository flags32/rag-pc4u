from haystack import Pipeline
from haystack.components.converters.txt import TextFileToDocument
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack_integrations.components.embedders.fastembed import FastembedSparseDocumentEmbedder
from haystack_integrations.components.embedders.ollama import OllamaDocumentEmbedder
from haystack.components.writers import DocumentWriter
from rag_pc4u.core.components import get_document_store
from rag_pc4u.core.config import settings
from rag_pc4u.core.custom_components.enricher import MetadataEnricher

def build_indexing_pipeline() -> Pipeline:
    """Assemble le pipeline d'ingestion hybride (Dense + Sparse)."""
    ds = get_document_store()
    pipeline = Pipeline()

    # 1. Extraction et Préparation du texte
    pipeline.add_component("converter", TextFileToDocument())
    pipeline.add_component("cleaner", DocumentCleaner())
    pipeline.add_component("splitter", DocumentSplitter(
        split_by="word",
        split_length=settings.chunk_size,
        split_overlap=settings.chunk_overlap
    ))

    # 2. Cloisonnement
    pipeline.add_component("enricher", MetadataEnricher())

    # 3. Vectorisation (Hybride)
    pipeline.add_component("dense_embedder", OllamaDocumentEmbedder(
        url=settings.ollama_host,
        model=settings.ollama_embed_model
    ))
    pipeline.add_component(
        "sparse_embedder",
        FastembedSparseDocumentEmbedder(
            model="Qdrant/bm25",
            providers=["CPUExecutionProvider"]
        )
    )

    # 4. Stockage
    pipeline.add_component("writer", DocumentWriter(document_store=ds))

    # --- CÂBLAGE DU GRAPHE ---
    pipeline.connect("converter.documents", "cleaner.documents")
    pipeline.connect("cleaner.documents", "splitter.documents")
    pipeline.connect("splitter.documents", "enricher.documents")

    # Pour le mode hybride, on chaîne les embedders pour que le même document reçoive à la fois son vecteur dense ET son vecteur sparse avant l'écriture.
    pipeline.connect("enricher.documents", "dense_embedder.documents")
    pipeline.connect("dense_embedder.documents", "sparse_embedder.documents")
    pipeline.connect("sparse_embedder.documents", "writer.documents")

    return pipeline