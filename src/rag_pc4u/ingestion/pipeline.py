"""Pipeline d'indexation Haystack — ciblé par collection."""
from haystack import Pipeline
from haystack.components.converters.txt import TextFileToDocument
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.routers import FileTypeRouter
from haystack.components.writers import DocumentWriter
from haystack.components.joiners import DocumentJoiner

from haystack_integrations.components.embedders.fastembed import (
    FastembedSparseDocumentEmbedder,
)
from haystack_integrations.components.embedders.ollama import OllamaDocumentEmbedder

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_haystack.converter import DoclingConverter
from docling.chunking import HybridChunker

from rag_pc4u.core.components import get_document_store
from rag_pc4u.core.config import settings
from rag_pc4u.core.custom_components.csv_converter import CSVRowToDocument
# Import unique depuis custom_components — aucune duplication
from rag_pc4u.core.custom_components.enricher import MetadataEnricher
from rag_pc4u.core.custom_components.extensionless import ExtensionlessToDocument


def _make_docling_converter() -> DoclingConverter:
    """Construit le convertisseur Docling pour les PDF."""
    pdf_pipeline_options = PdfPipelineOptions(do_ocr=False)
    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_pipeline_options)
        }
    )
    return DoclingConverter(
        converter=doc_converter,
        export_type="doc_chunks",
        chunker=HybridChunker(
            tokenizer="BAAI/bge-m3",
            max_tokens=settings.chunk_size,
        ),
    )


def build_indexing_pipeline(collection_name: str) -> Pipeline:
    """
    Construit le pipeline d'indexation ciblé sur une collection précise.

    Le document_store utilisé par le writer est instancié pour cette collection.
    Aucun paramètre client_id — l'isolation est physique (une collection = un espace).

    Args:
        collection_name : Nom de la collection Qdrant cible.
    """
    ds = get_document_store(collection_name)
    pipeline = Pipeline()

    # ── Routage et conversion ─────────────────────────────────────────────────
    # Ajout du type MIME pour le CSV
    pipeline.add_component(
        "router",
        FileTypeRouter(mime_types=["text/plain", "application/pdf", "text/markdown", "text/csv"]),
    )
    pipeline.add_component("pdf_converter", _make_docling_converter())
    pipeline.add_component("txt_converter", TextFileToDocument())
    pipeline.add_component("md_converter", TextFileToDocument())
    pipeline.add_component("extensionless_converter", ExtensionlessToDocument())

    # Ajout du composant CSV
    pipeline.add_component("csv_converter", CSVRowToDocument())

    # ── Joiners ───────────────────────────────────────────────────────────────
    pipeline.add_component("joiner_txt", DocumentJoiner(join_mode="concatenate"))
    pipeline.add_component("joiner_main", DocumentJoiner(join_mode="concatenate"))

    # ── Nettoyage et découpage ────────────────────────────────────────────────
    pipeline.add_component("cleaner", DocumentCleaner())
    pipeline.add_component(
        "splitter",
        DocumentSplitter(
            split_by="word",
            split_length=settings.chunk_size,
            split_overlap=settings.chunk_overlap,
        ),
    )

    # ── Enrichissement (file_path, file_name, date_added) ────────────────────
    pipeline.add_component("enricher", MetadataEnricher())

    # ── Embeddings hybrides ───────────────────────────────────────────────────
    pipeline.add_component(
        "dense_embedder",
        OllamaDocumentEmbedder(
            model=settings.ollama_embed_model,
            url=settings.ollama_host,
        ),
    )
    pipeline.add_component(
        "sparse_embedder",
        FastembedSparseDocumentEmbedder(model="Qdrant/bm25", parallel=None),
    )

    # ── Écriture dans la collection ───────────────────────────────────────────
    pipeline.add_component("writer", DocumentWriter(document_store=ds))

    # ── Câblage ───────────────────────────────────────────────────────────────

    # 1. Routage vers les convertisseurs (ajout du CSV)
    pipeline.connect("router.application/pdf", "pdf_converter.sources")
    pipeline.connect("router.text/plain", "txt_converter.sources")
    pipeline.connect("router.text/markdown", "md_converter.sources")
    pipeline.connect("router.text/csv", "csv_converter.sources")
    pipeline.connect("router.unclassified", "extensionless_converter.sources")

    # 2. Collecte des formats textuels purs (qui doivent être découpés)
    pipeline.connect("txt_converter.documents", "joiner_txt.documents")
    pipeline.connect("md_converter.documents", "joiner_txt.documents")
    pipeline.connect("extensionless_converter.documents", "joiner_txt.documents")

    # 3. Nettoyage et découpage de la branche texte
    pipeline.connect("joiner_txt.documents", "cleaner.documents")
    pipeline.connect("cleaner.documents", "splitter.documents")

    # 4. Fusion finale : chunks PDF + chunks texte + chunks CSV
    pipeline.connect("pdf_converter.documents", "joiner_main.documents")
    pipeline.connect("splitter.documents", "joiner_main.documents")
    # Le CSV rejoint le pipeline ici (bypass du splitter !)
    pipeline.connect("csv_converter.documents", "joiner_main.documents")

    # 5. Enrichissement → embeddings → stockage
    pipeline.connect("joiner_main.documents", "enricher.documents")
    pipeline.connect("enricher.documents", "dense_embedder.documents")
    pipeline.connect("dense_embedder.documents", "sparse_embedder.documents")
    pipeline.connect("sparse_embedder.documents", "writer.documents")

    return pipeline