"""Module de routes pour l'ingestion de données dans l'API RAG PC4U."""
from fastapi import APIRouter, UploadFile, File
import tempfile, shutil
from pathlib import Path
from fastapi import HTTPException
from rag_pc4u.ingestion.pipeline import build_indexing_pipeline

router = APIRouter()

ALLOWED_TYPES: dict[str, str] = {
    "text/plain": ".txt",
    "text/markdown": ".md",
    "application/pdf": ".pdf",
    "application/octet-stream": "",
}

@router.post("/ingest")
async def ingest_file(file: UploadFile = File(...), client_id: str = "client_demo"):
    """
    Ingère un fichier (TXT, MD ou PDF) dans Qdrant pour le client donné.

    Le fichier est écrit temporairement sur disque car les convertisseurs
    Haystack ont besoin d'un chemin fichier, pas d'un stream.
    """
    # 1. Valider le type MIME déclaré par le client
    content_type = file.content_type or ""
    # Normalise : "application/pdf; charset=..." → "application/pdf"
    mime_base = content_type.split(";")[0].strip()

    if mime_base not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Type de fichier non supporté : '{mime_base}'. "
                   f"Types acceptés : {', '.join(ALLOWED_TYPES)}",
        )

    suffix = ALLOWED_TYPES[mime_base]

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = Path(tmp.name)

        pipeline = build_indexing_pipeline()
        results = pipeline.run({
            "router": {"sources": [tmp_path]},
            "enricher": {"client_id": client_id},
        })

        return {
            "status": "ok",
            "filename": file.filename,
            "client_id": client_id,
            "documents_written": results["writer"]["documents_written"],
        }

    finally:
        # Nettoyage garanti même en cas d'exception
        if tmp_path and tmp_path.exists():
            tmp_path.unlink()
