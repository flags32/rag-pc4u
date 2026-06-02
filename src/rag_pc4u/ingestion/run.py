"""
Gestion incrémentale des documents.
Un document modifié remplace ses anciens chunks.
Un document supprimé est désindexé.
"""

import hashlib
import json
import sys
import structlog
from pathlib import Path

from rag_pc4u.core.logger_config import configure_logging
from rag_pc4u.core.components import get_document_store
from rag_pc4u.ingestion.pipeline import build_indexing_pipeline
from rag_pc4u.ingestion.sources import LocalDirectoryScanner

logger = structlog.get_logger(__name__)

# ✅ Fix #3 : ancré au dossier du script, indépendant du CWD
STATE_FILE = Path(__file__).parent / ".ingestion_state.json"


def _load_state() -> dict[str, str]:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, IOError):
            logger.warning("state.load_failed", path=str(STATE_FILE))
    return {}


def _save_state(state: dict[str, str]) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _delete_chunks_for_source(source_path: str, client_id: str) -> int:
    ds = get_document_store()
    filters = {
        "operator": "AND",
        "conditions": [
            {"field": "meta.client_id", "operator": "==", "value": client_id},
            {"field": "meta.file_path", "operator": "==", "value": source_path},
        ],
    }
    docs = ds.filter_documents(filters=filters)
    if docs:
        ds.delete_documents(document_ids=[d.id for d in docs])
        logger.info("incremental.deleted_chunks", source=source_path, count=len(docs))
    return len(docs)


def run_folder_ingestion(folder_path: str, client_id: str = "client_demo"):
    configure_logging()
    logger.info("Démarrage de l'ingestion incrémentale", path=folder_path, client_id=client_id)

    scanner = LocalDirectoryScanner(allowed_extensions=["", ".txt", ".md", ".pdf"])
    scan_results = scanner.run(directory_path=folder_path)
    current_files: list[Path] = scan_results["paths"]

    if not current_files:
        logger.warning("Aucun fichier trouvé dans le répertoire.")
        return

    previous_state = _load_state()
    new_state: dict[str, str] = {}
    files_to_index: list[Path] = []  # ✅ type hint clair
    files_deleted: list[str] = []

    current_paths_str = {str(p.resolve()) for p in current_files}

    for file_path in current_files:
        abs_path = str(file_path.resolve())
        current_hash = _sha256(file_path)
        new_state[abs_path] = current_hash

        previous_hash = previous_state.get(abs_path)
        if previous_hash is None:
            logger.info("incremental.new_file", path=abs_path)
            files_to_index.append(Path(abs_path))  # ✅ Fix #1 : Path, pas str
        elif previous_hash != current_hash:
            logger.info("incremental.modified_file", path=abs_path)
            _delete_chunks_for_source(abs_path, client_id)
            files_to_index.append(Path(abs_path))  # ✅ Fix #1 : Path, pas str
        else:
            logger.debug("incremental.unchanged_file", path=abs_path)

    for abs_path in previous_state:
        if abs_path not in current_paths_str:
            logger.info("incremental.deleted_file", path=abs_path)
            _delete_chunks_for_source(abs_path, client_id)
            files_deleted.append(abs_path)

    if files_to_index:
        logger.info("Indexation des fichiers modifiés/nouveaux", count=len(files_to_index))
        pipeline = build_indexing_pipeline()
        results = pipeline.run({
            "router": {"sources": files_to_index},
            "enricher": {"client_id": client_id},
        })
        logger.info(
            "Ingestion terminée",
            documents_ecrits=results.get("writer", {}).get("documents_written", 0),
            fichiers_supprimes=len(files_deleted),
        )
    else:
        logger.info("Aucun fichier à réindexer — base à jour.")

    _save_state(new_state)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_folder = sys.argv[1]
        client = sys.argv[2] if len(sys.argv) > 2 else "client_demo"
    else:
        target_folder = "/home/user/Documents/projet_rag/tests"
        client = "client_demo"

    run_folder_ingestion(target_folder, client_id=client)


