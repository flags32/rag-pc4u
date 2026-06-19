"""
Ingestion incrémentale par collection avec Workers Parallèles.
"""
import hashlib
import json
import sys
import os
import structlog
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from rag_pc4u.core.logger_config import configure_logging
from rag_pc4u.core.components import get_document_store
from rag_pc4u.core.tz_utils import now_paris_naive
from rag_pc4u.ingestion.pipeline import build_indexing_pipeline
from rag_pc4u.ingestion.sources import LocalDirectoryScanner

logger = structlog.get_logger(__name__)

class IngestionPendingFilesError(Exception):
    """
    Levée quand pipeline.run() échoue après que les anciens chunks d'un
    ou plusieurs fichiers ont déjà été supprimés de Qdrant. Porte la liste
    de ces fichiers pour que l'appelant (nextcloud_watcher.py) puisse la
    remonter explicitement au dashboard, au lieu de la perdre derrière un
    message d'erreur générique.
    """

    def __init__(self, message: str, pending_files: list[str]):
        super().__init__(message)
        self.pending_files = pending_files

def _state_file_for(collection_name: str) -> Path:
    safe_name = collection_name.replace("/", "_").replace(":", "_").replace(" ", "_")
    return Path(__file__).parent / f"fichier_injecter/.ingestion_state_{safe_name}.json"


def _load_state(collection_name: str) -> dict[str, str]:
    state_file = _state_file_for(collection_name)
    if state_file.exists():
        try:
            return json.loads(state_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, IOError):
            logger.warning("state.load_failed", path=str(state_file))
    return {}


def _save_state(collection_name: str, state: dict[str, str]) -> None:
    _state_file_for(collection_name).write_text(
        json.dumps(state, indent=2), encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _delete_chunks_for_source(source_path: str, collection_name: str) -> int:
    ds = get_document_store(collection_name)
    filters = {
        "field": "meta.file_path",
        "operator": "==",
        "value": source_path,
    }
    docs = ds.filter_documents(filters=filters)
    if docs:
        ds.delete_documents(document_ids=[d.id for d in docs])
        logger.info(
            "incremental.deleted_chunks",
            source=source_path,
            count=len(docs),
            collection=collection_name,
        )
    return len(docs)


def _worker_job(file_path: Path, collection_name: str, date_added: str) -> int:
    """
    TRAVAIL D'UN WORKER : Indexe un seul fichier de manière isolée.
    Cette fonction doit être globale pour être 'picklable' par ProcessPoolExecutor.
    chaque worker instancie son propre pipeline (sécurisé pour le multi-process).
    """
    try:
        # Initialisation locale du pipeline dans le worker
        pipeline = build_indexing_pipeline(collection_name)
        results = pipeline.run({
            "router": {"sources": [file_path]},
            "enricher": {"date_added": date_added},
        })
        return results.get("writer", {}).get("documents_written", 0)
    except Exception as e:
        # On remonte l'erreur au processus maître pour ne pas bloquer le reste de l'ingestion
        raise RuntimeError(f"Erreur lors du traitement de {file_path}: {str(e)}")


def run_folder_ingestion(folder_path: str, collection_name: str) -> None:
    """
    Indexe incrémentalement un dossier en parallélisant le calcul sur plusieurs cœurs.
    """
    configure_logging()
    logger.info(
        "Démarrage de l'ingestion incrémentale PARALLÈLE",
        path=folder_path,
        collection=collection_name,
    )

    # Ton scanner mis à jour prendra automatiquement en charge les nouvelles extensions
    scanner = LocalDirectoryScanner(
        allowed_extensions=[
            "", ".txt", ".md", ".pdf", ".csv",
            ".docx", ".pptx", ".xlsx", ".html",
            ".jpg", ".jpeg", ".png", ".tiff",
            ".json", ".xml"
        ]
    )
    current_files: list[Path] = scanner.run(directory_path=folder_path)["paths"]

    previous_state = _load_state(collection_name)
    new_state: dict[str, str] = {}
    files_to_index: list[Path] = []
    files_deleted: list[str] = []
    current_paths_str = {str(p.resolve()) for p in current_files}

    # [Maître] Analyse séquentielle rapide des hashes pour détecter les changements
    for file_path in current_files:
        abs_path = str(file_path.resolve())
        current_hash = _sha256(file_path)
        new_state[abs_path] = current_hash

        previous_hash = previous_state.get(abs_path)
        if previous_hash is None:
            logger.info("incremental.new_file", path=abs_path)
            files_to_index.append(Path(abs_path))
        elif previous_hash != current_hash:
            logger.info("incremental.modified_file", path=abs_path)
            _delete_chunks_for_source(abs_path, collection_name)
            files_to_index.append(Path(abs_path))
        else:
            new_state[abs_path] = previous_hash  # On garde l'ancien si inchangé

    for abs_path in previous_state:
        if abs_path not in current_paths_str:
            logger.info("incremental.deleted_file", path=abs_path)
            _delete_chunks_for_source(abs_path, collection_name)
            files_deleted.append(abs_path)

    # [Workers] Distribution du travail lourd (Docling, OCR, Embeddings)
    if files_to_index:
        # Configuration des cœurs : avec 56 cœurs, on peut allouer par exemple 16 ou 24 workers
        # pour l'ingestion sans saturer les entrées/sorties ou le serveur Ollama.
        max_workers = min(16, len(files_to_index))

        logger.info(
            "Distribution du calcul aux workers CPU",
            fichiers_a_traiter=len(files_to_index),
            workers_actifs=max_workers,
            collection=collection_name,
        )

        date_str = now_paris_naive().strftime("%Y-%m-%d %H:%M:%S")
        total_documents_ecrits = 0

        # Lancement de la mêlée parallèle
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # On soumet chaque fichier individuellement à la file d'attente des cœurs
            futures = {
                executor.submit(_worker_job, f, collection_name, date_str): f
                for f in files_to_index
            }

            # Récupération des résultats au fil de l'eau
            for future in as_completed(futures):
                file_path = futures[future]
                try:
                    written = future.result()
                    total_documents_ecrits += written
                    logger.info("Fichier indexé avec succès", path=str(file_path), chunks=written)
                except Exception as e:
                    logger.error("Échec de l'indexation d'un fichier", path=str(file_path), error=str(e))
                    # IMPORTANT : En cas d'échec sur un fichier, on le retire du new_state
                    # pour qu'il soit retenté au prochain scan.
                    new_state.pop(str(file_path.resolve()), None)

        logger.info(
            "Ingestion parallèle terminée",
            total_chunks_qdrant=total_documents_ecrits,
            fichiers_supprimes=len(files_deleted),
            collection=collection_name,
        )
    else:
        logger.info("Aucun fichier à réindexer — base à jour.", collection=collection_name)

    # [Maître] Sauvegarde finale sécurisée de l'état global
    _save_state(collection_name, new_state)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_folder = sys.argv[1]
        collection = sys.argv[2] if len(sys.argv) > 2 else "documents_default"
    else:
        target_folder = "/home/user/Documents/projet_rag/tests"
        collection = "documents_default"

    run_folder_ingestion(target_folder, collection_name=collection)