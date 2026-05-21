"""Module de routes pour l'ingestion de données dans l'API RAG PC4U."""
from fastapi import APIRouter, UploadFile, File
import tempfile, shutil
from pathlib import Path

from rag_pc4u.ingestion.pipeline import build_indexing_pipeline

router = APIRouter()


@router.post("/ingest")
async def ingest_file(file: UploadFile = File(...), client_id: str = "client_demo"):
    # Écriture temporaire sur disque car TextFileToDocument a besoin d'un path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    pipeline = build_indexing_pipeline()
    results = pipeline.run({
        "converter": {"sources": [tmp_path]},
        "enricher": {"client_id": client_id}
    })

    Path(tmp_path).unlink()  # nettoyage
    return {"documents_written": results["writer"]["documents_written"]}