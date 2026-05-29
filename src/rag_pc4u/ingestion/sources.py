"""Sources module Rag PC4U"""

import os
from pathlib import Path
from typing import List
from haystack import component
import structlog

logger = structlog.get_logger(__name__)


@component
class LocalDirectoryScanner:
    """
    Composant Haystack custom qui scanne un répertoire et renvoie les chemins de fichiers.
    Agit comme un "wrapper" pour les points de montage SMB/Nextcloud locaux.
    """

    def __init__(self, allowed_extensions: List[str] = None):
        self.allowed_extensions = allowed_extensions or ["",".txt", ".pdf", ".md"]

    @component.output_types(paths=List[Path])
    def run(self, directory_path: str):
        """Exécute le scan du répertoire."""
        path = Path(directory_path)
        if not path.exists() or not path.is_dir():
            logger.error("scanner.directory_not_found", path=str(path))
            return {"paths": []}

        file_paths = []
        for root, _, files in os.walk(path):
            for file in files:
                ext = Path(file).suffix.lower()
                if ext in self.allowed_extensions:
                    file_paths.append(Path(root) / file)

        logger.info("scanner.files_found", count=len(file_paths), path=str(path))
        return {"paths": file_paths}