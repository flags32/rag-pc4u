"""Composant Haystack custom : enrichissement des métadonnées avant indexation."""
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from haystack import component, Document

logger = logging.getLogger(__name__)

@component
class MetadataEnricher:
    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document], date_added: Optional[str] = None) -> dict:
        if not date_added:
            date_added = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        enriched = []
        for doc in documents:
            # 1. Extraction ultra-robuste du chemin source
            # - TextFileToDocument pose "source"
            # - CSVRowToDocument pose "file_path"
            # - Docling pose parfois ses métadonnées dans "origin" ou "dl_meta"
            raw_path = (
                doc.meta.get("file_path")
                or doc.meta.get("source")
                or (doc.meta.get("origin") or {}).get("filename")
                or (doc.meta.get("origin") or {}).get("uri")
                or (doc.meta.get("dl_meta") or {}).get("origin", {}).get("uri")
                or (doc.meta.get("dl_meta") or {}).get("origin", {}).get("filename")
                or ""
            )
            file_path = str(raw_path)

            # 2. Nettoyage du préfixe file:// s'il est ajouté par Docling
            if file_path.startswith("file://"):
                file_path = file_path[7:]

            # 3. Récupération du nom du fichier seul
            file_name = Path(file_path).name if file_path else "inconnu"

            if not file_path:
                warnings.warn(
                    f"MetadataEnricher: aucun chemin trouvé pour doc.id={doc.id}. "
                    f"meta reçu = {doc.meta!r}",
                    stacklevel=2,
                )
                logger.debug("MetadataEnricher — meta complet:\n%s", _pretty_meta(doc.meta))

            new_meta = {
                **doc.meta,
                "file_path": file_path,    # chemin complet → désindexation incrémentale
                "file_name": file_name,    # nom seul       → citation dans le prompt
                "date_added": date_added,
            }

            enriched.append(
                Document(
                    content=doc.content,
                    meta=new_meta,
                    id=doc.id,
                )
            )
        return {"documents": enriched}

    def __repr__(self):
        return "MetadataEnricher()"


def _pretty_meta(meta: dict) -> str:
    """Sérialisation lisible du meta pour les logs, sans crasher sur les types inconnus."""
    lines = []
    for k, v in meta.items():
        lines.append(f"  {k!r}: {v!r}")
    return "\n".join(lines) if lines else "  (vide)"