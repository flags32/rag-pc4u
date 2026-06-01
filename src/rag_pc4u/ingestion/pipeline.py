from haystack import Pipeline
from haystack.components.converters.txt import TextFileToDocument
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.routers import FileTypeRouter
from haystack.components.writers import DocumentWriter

from haystack_integrations.components.embedders.fastembed import (
    FastembedSparseDocumentEmbedder,
)
from haystack_integrations.components.embedders.ollama import OllamaDocumentEmbedder

# Docling
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TesseractOcrOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_haystack.converter import DoclingConverter
from docling.chunking import HybridChunker

from rag_pc4u.core.components import get_document_store
from rag_pc4u.core.config import settings
from rag_pc4u.core.custom_components.enricher import MetadataEnricher
from rag_pc4u.core.custom_components.extensionless import ExtensionlessToDocument


def _make_docling_converter() -> DoclingConverter:
    """
    Construit un DocumentConverter Docling avec Tesseract OCR activé.
    - do_ocr=True       : OCR activé sur les pages où le texte est absent/rare
    - lang=["auto"]     : détection automatique de la langue (fr, en, etc.)
    - force_full_page_ocr=False : OCR seulement là où c'est nécessaire (perf)
    """
    ocr_options = TesseractOcrOptions(lang=["auto"])

    pdf_pipeline_options = PdfPipelineOptions(
        do_ocr=True,
        force_full_page_ocr=False,   # True = force sur toutes les pages (PDFs 100% scannés)
        ocr_options=ocr_options,
    )

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pdf_pipeline_options,
            )
        }
    )

    return DoclingConverter(
        converter=doc_converter,
        export_type="doc_chunks",          # chunking natif Docling (respecte le layout)
        chunker=HybridChunker(             # chunker hybride : sémantique + structure
            tokenizer=settings.ollama_embed_model,
            max_tokens=settings.chunk_size,
        ),
    )


def build_indexing_pipeline() -> Pipeline:
    """
    Pipeline d'ingestion hybride (Dense + Sparse).

    Branches :
      - application/pdf  → DoclingConverter (layout + Tesseract OCR + chunking intégré)
      - text/plain       → TextFileToDocument → cleaner → splitter
      - unclassified     → ExtensionlessToDocument → cleaner → splitter
    Les trois branches convergent vers enricher → embedders → writer.
    """
    ds = get_document_store()
    pipeline = Pipeline()

    # ── Routage ──────────────────────────────────────────────────────────────
    pipeline.add_component(
        "router",
        FileTypeRouter(mime_types=["text/plain", "application/pdf"]),
    )

    # ── Branche PDF : Docling ─────────────────────────────────────────────────
    # Docling chunk nativement → pas besoin de cleaner/splitter pour les PDFs
    pipeline.add_component("pdf_converter", _make_docling_converter())

    # ── Branche TXT ──────────────────────────────────────────────────────────
    pipeline.add_component("txt_converter", TextFileToDocument())

    # ── Branche sans extension ────────────────────────────────────────────────
    pipeline.add_component("extensionless_converter", ExtensionlessToDocument())

    # ── Nettoyage + Split (txt & sans extension uniquement) ───────────────────
    pipeline.add_component("cleaner", DocumentCleaner())
    pipeline.add_component(
        "splitter",
        DocumentSplitter(
            split_by="word",
            split_length=settings.chunk_size,
            split_overlap=settings.chunk_overlap,
        ),
    )

    # ── Enrichissement tenant ─────────────────────────────────────────────────
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

    # ── Stockage ──────────────────────────────────────────────────────────────
    pipeline.add_component("writer", DocumentWriter(document_store=ds))

    # ── CÂBLAGE ───────────────────────────────────────────────────────────────

    # Routage vers les convertisseurs
    pipeline.connect("router.application/pdf", "pdf_converter.paths")
    pipeline.connect("router.text/plain", "txt_converter.sources")
    pipeline.connect("router.unclassified", "extensionless_converter.sources")

    # PDF → directement vers enricher (Docling a déjà chunké)
    pipeline.connect("pdf_converter.documents", "enricher.documents")

    # TXT + sans extension → cleaner → splitter → enricher
    pipeline.connect("txt_converter.documents", "cleaner.documents")
    pipeline.connect("extensionless_converter.documents", "cleaner.documents")
    pipeline.connect("cleaner.documents", "splitter.documents")
    pipeline.connect("splitter.documents", "enricher.documents")

    # Tronc commun
    pipeline.connect("enricher.documents", "dense_embedder.documents")
    pipeline.connect("dense_embedder.documents", "sparse_embedder.documents")
    pipeline.connect("sparse_embedder.documents", "writer.documents")

    return pipeline