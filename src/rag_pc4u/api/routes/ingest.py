"""Module de routes pour l'ingestion de données dans l'API RAG PC4U."""
import tempfile
import shutil
from pathlib import Path
import structlog

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from rag_pc4u.ingestion.pipeline import build_indexing_pipeline
from rag_pc4u.core.config import settings

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["Ingest"])

@router.post("/ingest")
async def ingest_file(
    file: UploadFile = File(...),
    collection_name: str = Form(None)
):
    """
    Ingère un fichier dans Qdrant pour la collection donnée.
    Supporte PDF, TXT, MD, CSV, et les fichiers sans extension.
    """
    # 1. Fallback sur la collection par défaut
    if not collection_name:
        collection_name = settings.default_collection

    # 2. Sécurité basique : on évite d'envoyer des vidéos/images au TextConverter
    content_type = file.content_type or ""
    mime_base = content_type.split(";")[0].strip()

    if mime_base.startswith(("image/", "video/", "audio/")):
        raise HTTPException(
            status_code=415,
            detail=f"Les médias ({mime_base}) ne sont pas supportés par le RAG."
        )

    # 3. Extraction de l'extension originale (peut être vide pour les fichiers sans extension)
    original_suffix = Path(file.filename).suffix.lower()

    tmp_path = None
    try:
        # On sauvegarde le fichier avec son extension d'origine.
        # Si pas d'extension, suffix="" -> Haystack le classera en 'unclassified'
        with tempfile.NamedTemporaryFile(delete=False, suffix=original_suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = Path(tmp.name)

        logger.info(
            "ingestion.start",
            file=file.filename,
            collection=collection_name,
            mime_type=mime_base
        )

        # 4. Appel au pipeline Haystack ciblé
        pipeline = build_indexing_pipeline(collection_name=collection_name)

        results = pipeline.run({
            "router": {"sources": [tmp_path]}
        })

        docs_written = results.get("writer", {}).get("documents_written", 0)

        return {
            "status": "ok",
            "filename": file.filename,
            "collection": collection_name,
            "documents_written": docs_written,
        }

    except Exception as e:
        logger.error("ingestion.error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Erreur d'ingestion : {str(e)}")

    finally:
        # Nettoyage garanti
        if tmp_path and tmp_path.exists():
            tmp_path.unlink()