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

# Docling
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
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
- do_ocr=False       : OCR désactivé par défaut
- lang=["auto"]     : détection automatique de la langue (fr, en, etc.)
- force_full_page_ocr=False : OCR seulement là où c'est nécessaire (perf)
"""

    #ocr_options = TesseractOcrOptions(lang=["auto"])

    pdf_pipeline_options = PdfPipelineOptions(
        do_ocr=False,
        #force_full_page_ocr=False,
        #ocr_options=ocr_options,
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
        export_type="doc_chunks",
        chunker=HybridChunker(
            tokenizer="BAAI/bge-m3",  # identifiant HuggingFace correct
            max_tokens=settings.chunk_size,
        ),
    )


def build_indexing_pipeline() -> Pipeline:
    ds = get_document_store()
    pipeline = Pipeline()

    # ── Routage ──────────────────────────────────────────────────────────────
    pipeline.add_component(
        "router",
        FileTypeRouter(mime_types=["text/plain", "application/pdf", "text/markdown"]),
    )

    # ── Branche PDF : Docling ─────────────────────────────────────────────────
    pipeline.add_component("pdf_converter", _make_docling_converter())

    # ── Branche TXT ───────────────────────────────────────────────────────────
    pipeline.add_component("txt_converter", TextFileToDocument())

    # ── Branche MD ────────────────────────────────────────────────────────────
    pipeline.add_component("md_converter", TextFileToDocument())

    # ── Branche sans extension ────────────────────────────────────────────────
    pipeline.add_component("extensionless_converter", ExtensionlessToDocument())

    # ── Joiners ───────────────────────────────────────────────────────────────
    pipeline.add_component("joiner_txt", DocumentJoiner(join_mode="concatenate"))
    pipeline.add_component("joiner_main", DocumentJoiner(join_mode="concatenate"))

    # ── Nettoyage + Split ─────────────────────────────────────────────────────
    pipeline.add_component("cleaner", DocumentCleaner())
    pipeline.add_component(
        "splitter",
        DocumentSplitter(
            split_by="word",
            split_length=settings.chunk_size,
            split_overlap=settings.chunk_overlap,
        ),
    )

    # ── Enrichissement ────────────────────────────────────────────────────────
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

    # ── CÂBLAGE CORRIGÉ ───────────────────────────────────────────────────────

    # 1. Routage vers les convertisseurs (1 seule connexion par port d'entrée d'un composant)
    pipeline.connect("router.application/pdf", "pdf_converter.sources")
    pipeline.connect("router.text/plain", "txt_converter.sources")
    pipeline.connect("router.text/markdown", "md_converter.sources")
    pipeline.connect("router.unclassified", "extensionless_converter.sources")

    # 2. Le Joiner accepte plusieurs connexions sur son port unique 'documents'
    pipeline.connect("txt_converter.documents", "joiner_txt.documents")
    pipeline.connect("md_converter.documents", "joiner_txt.documents")
    pipeline.connect("extensionless_converter.documents", "joiner_txt.documents")

    # 3. Traitement de la branche de texte unifiée
    pipeline.connect("joiner_txt.documents", "cleaner.documents")
    pipeline.connect("cleaner.documents", "splitter.documents")

    # 4. Fusion finale : Chunks PDF (via Docling) + Chunks textuels
    pipeline.connect("pdf_converter.documents", "joiner_main.documents")
    pipeline.connect("splitter.documents", "joiner_main.documents")

    # 5. Tronc commun
    pipeline.connect("joiner_main.documents", "enricher.documents")
    pipeline.connect("enricher.documents", "dense_embedder.documents")
    pipeline.connect("dense_embedder.documents", "sparse_embedder.documents")
    pipeline.connect("sparse_embedder.documents", "writer.documents")

    return pipeline
