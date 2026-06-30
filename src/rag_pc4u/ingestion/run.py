"""
Ingestion incrémentale par collection avec Workers Parallèles.
Design sécurisé : Nettoyage amont (Delete Old) -> Distribution (Write New)
"""
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import structlog

from rag_pc4u.core.config import settings
from rag_pc4u.core.logger_config import configure_logging
from rag_pc4u.core.components import get_document_store
from rag_pc4u.core.tz_utils import now_paris_naive
from rag_pc4u.ingestion.pipeline import build_indexing_pipeline
from rag_pc4u.ingestion.sources import LocalDirectoryScanner

logger = structlog.get_logger(__name__)

class IngestionPendingFilesError(Exception):
    """
    Exception levée lorsque des fichiers échouent lors de l'indexation parallèle.
    Capturée par nextcloud_watcher.py pour notifier le dashboard.
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


def deindex_file(local_path: str, collection_name: str) -> int:
    """
    Désindexe un fichier individuel de Qdrant et met à jour l'état local.

    - Supprime tous les chunks de ce fichier dans la collection Qdrant.
    - Retire le fichier du cache local sur disque.
    - Retire l'entrée de l'état d'ingestion (.ingestion_state_<collection>.json)
      pour que le fichier soit retecté comme nouveau lors du prochain cycle
      si jamais il réapparaît côté Nextcloud.

    Args:
        local_path      : Chemin absolu du fichier dans le cache local.
        collection_name : Collection Qdrant cible.

    Returns:
        Nombre de chunks Qdrant supprimés.
    """
    count = _delete_chunks_for_source(local_path, collection_name)

    # Suppression du fichier dans le cache local
    p = Path(local_path)
    if p.exists():
        try:
            p.unlink()
            logger.info("deindex.local_file_removed", path=local_path)
        except Exception as e:
            logger.warning("deindex.local_file_removal_failed", path=local_path, error=str(e))

    # Retrait de l'état d'ingestion pour forcer la redétection au prochain cycle
    state = _load_state(collection_name)
    if local_path in state:
        state.pop(local_path)
        _save_state(collection_name, state)
        logger.info("deindex.state_entry_removed", path=local_path)

    return count


def deindex_collection(collection_name: str) -> int:
    """
    Désindexe TOUS les fichiers d'une collection et supprime la collection
    Qdrant elle-même.

    Utilisé par le workflow "Désindexer tout" du dashboard :
      1. Compte les chunks de la collection (pour info/retour).
      2. Supprime la collection Qdrant (drop physique, atomique).
      3. Supprime le fichier d'état d'ingestion local.

    Args:
        collection_name : Collection Qdrant à vider et supprimer.

    Returns:
        Nombre de chunks supprimés.
    """
    ds = get_document_store(collection_name)

    """
    Erreur 3
    Compte les chunks AVANT le drop, uniquement pour le retour/les logs.
    On évite volontairement tout filter_documents()/delete_documents() sur
    l'ensemble de la collection : ça charge potentiellement des milliers de
    documents (contenu + embeddings) en mémoire puis envoie un delete massif
    en un seul appel réseau, ce qui provoque des timeouts sur les grosses
    collections (cf. deindex.collection_chunks_failed). Comme la collection
    va être droppée physiquement juste après, ce nettoyage document-par-
    document est redondant : delete_collection supprime tout d'un coup.
    """
    try:
        count = ds.count_documents()
    except Exception as e:
        logger.warning(
            "deindex.collection_count_failed",
            collection=collection_name,
            error=str(e),
        )
        count = 0

    # Suppression physique de la collection dans Qdrant (opération atomique
    # côté Qdrant, ne nécessite pas de supprimer les points un par un avant).
    try:
        ds.client.delete_collection(collection_name)
        logger.info(
            "deindex.collection_dropped",
            collection=collection_name,
            count=count,
        )
    except Exception as e:
        logger.error(
            "deindex.collection_drop_failed",
            collection=collection_name,
            error=str(e),
        )
        raise

    # Suppression de l'état d'ingestion local
    state_file = _state_file_for(collection_name)
    if state_file.exists():
        try:
            state_file.unlink()
            logger.info("deindex.state_file_removed", path=str(state_file))
        except Exception as e:
            logger.warning("deindex.state_file_removal_failed", error=str(e))

    return count


def count_indexed_chunks(collection_name: str) -> int:
    """
    Retourne le nombre de chunks actuellement indexés dans une collection.
    Utilisé par api.py pour bloquer la suppression d'un mapping tant que
    des données sont encore présentes dans Qdrant.
    """
    try:
        ds = get_document_store(collection_name)
        return len(ds.filter_documents())
    except Exception:
        # La collection n'existe pas encore ou Qdrant est inaccessible —
        # on retourne 0 pour ne pas bloquer inutilement l'utilisateur.
        return 0


def _worker_job(file_path: Path, collection_name: str, date_added: str) -> int:
    """
    JOB ATOMIQUE DU WORKER : Construit son propre pipeline isolé
    et indexe un unique fichier reçu du Maître.
    """
    try:
        pipeline = build_indexing_pipeline(collection_name)
        results = pipeline.run({
            "router": {"sources": [file_path]},
            "enricher": {"date_added": date_added},
        })
        return results.get("writer", {}).get("documents_written", 0)
    except Exception as e:
        raise RuntimeError(f"Erreur lors du traitement de {file_path}: {str(e)}")


def run_folder_ingestion(folder_path: str, collection_name: str) -> Dict[str, Any]:
    """
    Indexe de manière incrémentale et parallèle un dossier local.
    Garantit la cohérence des données en nettoyant Qdrant avant réindexation.
    """
    configure_logging()
    logger.info(
        "Démarrage de l'ingestion incrémentale PARALLÈLE (V2-Sécurisée)",
        path=folder_path,
        collection=collection_name,
    )

    scanner = LocalDirectoryScanner(
        allowed_extensions=[
            "", ".txt", ".md", ".pdf", ".csv", ".docx", ".pptx", ".xlsx", ".html",
            # Images
            ".jpg", ".jpeg", ".png", ".tiff",
            # Données structurées
            ".json", ".xml",
            # Audio
            ".mp3", ".wav", ".m4a", ".aac", ".ogg", ".opus", ".wma", ".flac",
            # Vidéo
            ".mp4", ".mov"
        ]
    )
    current_files: list[Path] = scanner.run(directory_path=folder_path)["paths"]

    previous_state = _load_state(collection_name)
    new_state: dict[str, str] = {}

    files_to_index: list[Path] = []
    files_deleted: list[str] = []
    pending_files: list[str] = []

    current_paths_str = {str(p.resolve()) for p in current_files}

    # [MAÎTRE] Phase 1 : Scan, détection et NETTOYAGE IMMÉDIAT
    for file_path in current_files:
        abs_path = str(file_path.resolve())
        current_hash = _sha256(file_path)
        previous_hash = previous_state.get(abs_path)

        if previous_hash is None:
            # Nouveau fichier rencontré
            logger.info("incremental.new_file", path=abs_path)
            files_to_index.append(Path(abs_path))
            new_state[abs_path] = current_hash

        elif previous_hash != current_hash:
            # Fichier modifié -> LOGIQUE SAINE : On purge l'ancien contenu tout de suite
            logger.info("incremental.modified_file", path=abs_path)
            _delete_chunks_for_source(abs_path, collection_name)
            files_to_index.append(Path(abs_path))
            new_state[abs_path] = current_hash

        else:
            # Fichier inchangé
            new_state[abs_path] = previous_hash

    # Purge des fichiers supprimés du disque
    for abs_path in previous_state:
        if abs_path not in current_paths_str:
            logger.info("incremental.deleted_file", path=abs_path)
            _delete_chunks_for_source(abs_path, collection_name)
            files_deleted.append(abs_path)

    total_documents_ecrits = 0

    # [WORKERS] Phase 2 : Distribution de la charge d'écriture
    if files_to_index:
        # Configuration dynamique du nombre de Workers
        max_workers = getattr(settings, "max_workers", 24)
        max_workers = min(max_workers, len(files_to_index))

        logger.info(
            "Distribution des tâches aux workers",
            fichiers_total=len(files_to_index),
            workers_alloues=max_workers,
        )

        date_str = now_paris_naive().strftime("%Y-%m-%d %H:%M:%S")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_worker_job, f, collection_name, date_str): f
                for f in files_to_index
            }

            for future in as_completed(futures):
                file_path = futures[future]
                abs_path_str = str(file_path.resolve())

                try:
                    written = future.result()
                    total_documents_ecrits += written
                    logger.info("Fichier indexé avec succès", path=abs_path_str, chunks=written)
                except Exception as e:
                    logger.error("Échec de l'indexation", path=abs_path_str, error=str(e))
                    pending_files.append(abs_path_str)

                    # Gestion du Retry au prochain cycle :
                    # On restaure l'ancien état/hash pour forcer la détection de modification au prochain scan
                    if abs_path_str in previous_state:
                        new_state[abs_path_str] = previous_state[abs_path_str]
                    else:
                        new_state.pop(abs_path_str, None)

    # Enregistrement de l'état global
    _save_state(collection_name, new_state)

    logger.info(
        "Fin de la session d'ingestion",
        chunks_ajoutes=total_documents_ecrits,
        fichiers_purges=len(files_deleted),
        fichiers_en_erreur=len(pending_files),
    )

    # Si des fichiers ont échoué, on lève l'exception dédiée attendue par le Watcher
    if pending_files:
        raise IngestionPendingFilesError(
            f"{len(pending_files)} fichier(s) n'ont pas pu être indexés.",
            pending_files=pending_files
        )

    # CONTRAT REMPLI : Le dictionnaire de statistiques attendu par nextcloud_watcher.py
    return {
        "fichiers_traites": len(files_to_index),
        "fichiers_supprimes": len(files_deleted),
        "total_chunks_ecrits": total_documents_ecrits,
        "fichiers_en_attente": pending_files
    }


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_folder = sys.argv[1]
        collection = sys.argv[2] if len(sys.argv) > 2 else "documents_default"
    else:
        target_folder = "/home/user/Documents/projet_rag/tests"
        collection = "documents_default"

    try:
        result = run_folder_ingestion(target_folder, collection_name=collection)
        print(f"Succès : {result}")
    except IngestionPendingFilesError as e:
        print(f"Erreur d'ingestion détectée : {e.pending_files}")