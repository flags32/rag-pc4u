"""Run module Rag PC4U"""

import sys
import structlog
from pathlib import Path
from rag_pc4u.core.logging import configure_logging
from rag_pc4u.ingestion.pipeline import build_indexing_pipeline
from rag_pc4u.ingestion.sources import LocalDirectoryScanner  # On importe ton scanner

logger = structlog.get_logger(__name__)

def run_folder_ingestion(folder_path: str, client_id: str = "client_demo"):
    configure_logging()
    logger.info("Démarrage du scanner de répertoire...", path=folder_path)

    # 1. Utiliser le scanner pour lister dynamiquement les fichiers du dossier
    scanner = LocalDirectoryScanner(allowed_extensions=["", ".txt", ".md"])
    scan_results = scanner.run(directory_path=folder_path)
    files_to_ingest = scan_results["paths"]

    if not files_to_ingest:
        logger.warning("Aucun fichier valide trouvé dans le répertoire spécifié.")
        return

    logger.info("Fichiers trouvés, démarrage du pipeline...", total_files=len(files_to_ingest))

    # 2. Instancier et lancer le pipeline Haystack
    pipeline = build_indexing_pipeline()
    results = pipeline.run({
        "converter": {"sources": files_to_ingest},
        "enricher": {"client_id": client_id}
    })

    logger.info("Ingestion terminée avec succès !", documents_ecrits=results["writer"]["documents_written"])

if __name__ == "__main__":
    # Permet de passer le dossier en argument dans le terminal, ex: python run.py /chemin/mon_dossier
    if len(sys.argv) > 1:
        target_folder = sys.argv[1]
    else:
        # Chemin par défaut si tu ne mets pas d'argument
        target_folder = "/home/user/Documents/projet_rag/tests"

    run_folder_ingestion(target_folder)