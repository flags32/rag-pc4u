"""Pipeline d'indexation Haystack — ciblé par collection."""
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from haystack import Pipeline, Document, component
from haystack.components.converters.txt import TextFileToDocument
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.routers import FileTypeRouter
from haystack.components.writers import DocumentWriter
from haystack.components.joiners import DocumentJoiner
from haystack.dataclasses import ByteStream

from haystack_integrations.components.embedders.fastembed import (
    FastembedSparseDocumentEmbedder,
)
from haystack_integrations.components.embedders.ollama import OllamaDocumentEmbedder

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_haystack.converter import DoclingConverter
from docling.chunking import HybridChunker
from transformers import AutoTokenizer

from rag_pc4u.core.components import get_document_store
from rag_pc4u.core.config import settings
from rag_pc4u.core.custom_components.csv_converter import CSVRowToDocument
# Import unique depuis custom_components — aucune duplication
from rag_pc4u.core.custom_components.enricher import MetadataEnricher
from rag_pc4u.core.custom_components.extensionless import ExtensionlessToDocument

logger = logging.getLogger(__name__)

# Chemin local du cache HuggingFace (cohérent avec docker-compose.yml)
_HF_CACHE = Path(os.environ.get("HF_HOME", "/root/.cache/hf_cache"))
_BGE_M3_LOCAL = _HF_CACHE / "hub" / "models--BAAI--bge-m3"


@lru_cache(maxsize=1)
def _get_bge_tokenizer() -> AutoTokenizer:
    """
    Charge le tokenizer BAAI/bge-m3 depuis le cache local uniquement.
    Mis en cache pour n'être instancié qu'une seule fois.

    HF_HUB_OFFLINE=1 est positionné dans docker-compose, mais on force
    local_files_only=True ici aussi pour être explicite et fonctionner
    même hors Docker (ex: dev local).
    """
    # Cherche d'abord dans le snapshot le plus récent du cache
    snapshots_dir = _BGE_M3_LOCAL / "snapshots"
    if snapshots_dir.exists():
        snapshots = sorted(snapshots_dir.iterdir(), reverse=True)
        if snapshots:
            return AutoTokenizer.from_pretrained(
                str(snapshots[0]),
                local_files_only=True,
            )

    # Fallback : laisser HF chercher dans tout le cache via le nom du modèle
    # (fonctionne si HF_HOME est bien défini et HF_HUB_OFFLINE=1)
    return AutoTokenizer.from_pretrained(
        "BAAI/bge-m3",
        local_files_only=True,
        cache_dir=str(_HF_CACHE),
    )


class PatchedDoclingConverter(DoclingConverter):
    """
    Sous-classe de DoclingConverter qui garantit que file_path est toujours
    présent dans doc.meta après conversion.

    Docling avec export_type="doc_chunks" ne garantit pas que dl_meta.origin.filename
    est renseigné. On intercepte le résultat et on injecte file_path depuis la
    source d'origine selon trois niveaux de fallback :
      1. dl_meta.origin.filename / uri  — ce que Docling pose normalement
      2. binary_hash                    — si Docling a hashé le contenu
      3. source unique                  — si un seul fichier a été passé
    """

    @component.output_types(documents=List[Document])
    def run(
        self,
        sources: List[Union[str, Path, ByteStream]],
        meta: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    ) -> Dict[str, List[Document]]:
        result = super().run(sources=sources, meta=meta)
        docs: List[Document] = result.get("documents", [])

        source_paths = _extract_source_paths(sources)

        for doc in docs:
            # Si file_path est déjà là (ex: version future de docling), on ne touche pas
            if doc.meta.get("file_path"):
                continue

            # 1. Tenter de récupérer le chemin depuis dl_meta
            dl_origin = (doc.meta.get("dl_meta") or {}).get("origin") or {}
            path_from_docling = dl_origin.get("filename") or dl_origin.get("uri") or ""
            if path_from_docling.startswith("file://"):
                path_from_docling = path_from_docling[7:]

            if path_from_docling:
                doc.meta["file_path"] = path_from_docling
                continue

            # 2. Fallback via binary_hash
            binary_hash = dl_origin.get("binary_hash")
            if binary_hash and binary_hash in source_paths:
                doc.meta["file_path"] = source_paths[binary_hash]
                continue

            # 3. Dernier recours : source unique
            if len(source_paths) == 1:
                doc.meta["file_path"] = next(iter(source_paths.values()))
            else:
                logger.warning(
                    "PatchedDoclingConverter: impossible de déterminer file_path "
                    "pour doc.id=%s (meta=%r)", doc.id, doc.meta
                )

        return {"documents": docs}


def _extract_source_paths(
    sources: List[Union[str, Path, ByteStream]],
) -> Dict[Any, str]:
    """
    Construit un dict {clé → chemin_str} à partir des sources passées à Docling.
    La clé est le binary_hash pour un ByteStream, ou le chemin str pour un Path.
    """
    result: Dict[Any, str] = {}
    for src in sources:
        if isinstance(src, (str, Path)):
            p = str(src)
            result[p] = p
        elif isinstance(src, ByteStream):
            fp = (src.meta or {}).get("file_path", "")
            if fp:
                try:
                    result[hash(src.data)] = str(fp)
                except Exception:
                    pass
    return result


def _make_docling_converter() -> PatchedDoclingConverter:
    pdf_pipeline_options = PdfPipelineOptions(do_ocr=False)
    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_pipeline_options)
        }
    )
    return PatchedDoclingConverter(
        converter=doc_converter,
        export_type="doc_chunks",
        chunker=HybridChunker(
            # On passe l'objet tokenizer directement — Docling ne touche pas
            # à HuggingFace Hub, aucune requête réseau possible.
            tokenizer=_get_bge_tokenizer(),
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
    # store_full_path=True est OBLIGATOIRE ici : sans ce paramètre, les
    # versions récentes de Haystack ne stockent QUE le nom du fichier dans
    # meta.file_path (pas le chemin absolu complet). run.py et
    # nextcloud_watcher.py filtrent et suppriment les anciens chunks en
    # comparant le chemin absolu complet — si seul le nom de fichier est
    # stocké, ce filtre ne trouve jamais de correspondance, les anciens
    # chunks ne sont jamais supprimés, et chaque modification ajoute un
    # doublon au lieu de remplacer l'ancien contenu.
    pipeline.add_component("txt_converter", TextFileToDocument(store_full_path=True))
    pipeline.add_component("md_converter", TextFileToDocument(store_full_path=True))
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