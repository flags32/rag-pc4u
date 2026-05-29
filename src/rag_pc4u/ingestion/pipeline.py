from haystack import Pipeline
from haystack.components.converters import PyPDFToDocument
from haystack.components.converters.txt import TextFileToDocument
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.routers import FileTypeRouter
from haystack_integrations.components.embedders.fastembed import FastembedSparseDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack_integrations.components.embedders.ollama import OllamaDocumentEmbedder

from rag_pc4u.core.components import get_document_store
from rag_pc4u.core.config import settings
from rag_pc4u.core.custom_components.enricher import MetadataEnricher


def build_indexing_pipeline() -> Pipeline:
    """Assemble le pipeline d'ingestion hybride (Dense + Sparse)."""
    ds = get_document_store()
    pipeline = Pipeline()

    # 1. Extraction et Préparation du texte
    pipeline.add_component("router", FileTypeRouter(mime_types=["text/plain", "application/pdf"]))
    pipeline.add_component("txt_converter", TextFileToDocument())
    pipeline.add_component("pdf_converter", PyPDFToDocument())
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
        model=settings.ollama_embed_model,
        url=settings.ollama_host
    ))

    pipeline.add_component(
        "sparse_embedder",
        FastembedSparseDocumentEmbedder(
            model="Qdrant/bm25",
            parallel=None
        )
    )

    # 4. Stockage
    pipeline.add_component("writer", DocumentWriter(document_store=ds))

    # --- CÂBLAGE DU GRAPHE ---
    # On connecte le routeur aux bons convertisseurs selon le type de fichier
    pipeline.connect("router.text/plain", "txt_converter.sources")
    pipeline.connect("router.application/pdf", "pdf_converter.sources")
    pipeline.connect("router.unclassified", "txt_converter.sources")

    # On redirige la sortie des deux convertisseurs vers le cleaner commun
    pipeline.connect("txt_converter.documents", "cleaner.documents")
    pipeline.connect("pdf_converter.documents", "cleaner.documents")

    # Suite du traitement identique
    pipeline.connect("cleaner.documents", "splitter.documents")
    pipeline.connect("splitter.documents", "enricher.documents")
    pipeline.connect("enricher.documents", "dense_embedder.documents")
    pipeline.connect("dense_embedder.documents", "sparse_embedder.documents")
    pipeline.connect("sparse_embedder.documents", "writer.documents")

    return pipeline