"""
Ingestion incrémentale par collection.

Un document nouveau → indexé.
Un document modifié → anciens chunks supprimés, puis réindexé.
Un document supprimé → désindexé.

Usage : python -m rag_pc4u.ingestion.run <dossier> <nom_collection>
"""
import hashlib
import json
import sys
import structlog
from datetime import datetime
from pathlib import Path

from rag_pc4u.core.logger_config import configure_logging
from rag_pc4u.core.components import get_document_store
from rag_pc4u.ingestion.pipeline import build_indexing_pipeline
from rag_pc4u.ingestion.sources import LocalDirectoryScanner

logger = structlog.get_logger(__name__)


# Gestion de l'état par collection

def _state_file_for(collection_name: str) -> Path:
    """
    Un fichier d'état distinct par collection.
    Empêche les états de deux collections de se mélanger si on lance
    l'ingestion sur plusieurs dossiers en parallèle.
    """
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


# ── Utilitaires ───────────────────────────────────────────────────────────────

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _delete_chunks_for_source(source_path: str, collection_name: str) -> int:
    """
    Supprime dans la collection tous les chunks issus d'un fichier source.
    L'isolation est déjà assurée par la collection — pas besoin de filtrer
    par client_id.
    """
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


# ── Point d'entrée principal ──────────────────────────────────────────────────

def run_folder_ingestion(folder_path: str, collection_name: str) -> None:
    """
    Indexe incrémentalement un dossier vers une collection Qdrant.

    Args:
        folder_path     : Chemin du dossier local à indexer.
        collection_name : Nom de la collection Qdrant cible.
    """
    configure_logging()
    logger.info(
        "Démarrage de l'ingestion incrémentale",
        path=folder_path,
        collection=collection_name,
    )

    scanner = LocalDirectoryScanner(allowed_extensions=["", ".txt", ".md", ".pdf", ".csv"])
    current_files: list[Path] = scanner.run(directory_path=folder_path)["paths"]

    if not current_files:
        logger.warning("Aucun fichier trouvé dans le répertoire.")
        return

    previous_state = _load_state(collection_name)
    new_state: dict[str, str] = {}
    files_to_index: list[Path] = []
    files_deleted: list[str] = []
    current_paths_str = {str(p.resolve()) for p in current_files}

    # Détection des fichiers nouveaux ou modifiés
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
            logger.debug("incremental.unchanged_file", path=abs_path)

    # Détection des fichiers supprimés
    for abs_path in previous_state:
        if abs_path not in current_paths_str:
            logger.info("incremental.deleted_file", path=abs_path)
            _delete_chunks_for_source(abs_path, collection_name)
            files_deleted.append(abs_path)

    # Indexation
    if files_to_index:
        logger.info(
            "Indexation des fichiers modifiés/nouveaux",
            count=len(files_to_index),
            collection=collection_name,
        )
        pipeline = build_indexing_pipeline(collection_name)
        results = pipeline.run({
            "router": {"sources": files_to_index},
            # date_added transmis à MetadataEnricher — horodatage cohérent
            # pour tous les chunks d'une même session d'ingestion
            "enricher": {"date_added": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
        })
        logger.info(
            "Ingestion terminée",
            documents_ecrits=results.get("writer", {}).get("documents_written", 0),
            fichiers_supprimes=len(files_deleted),
            collection=collection_name,
        )
    else:
        logger.info("Aucun fichier à réindexer — base à jour.", collection=collection_name)

    _save_state(collection_name, new_state)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_folder = sys.argv[1]
        collection = sys.argv[2] if len(sys.argv) > 2 else "documents_default"
    else:
        target_folder = "/home/user/Documents/projet_rag/tests"
        collection = "documents_default"

    run_folder_ingestion(target_folder, collection_name=collection)