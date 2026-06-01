"""
Composant Haystack pour lire les fichiers sans extension.
Tente UTF-8, puis Latin-1 en fallback — couvre l'essentiel des fichiers texte.
"""
from pathlib import Path
from typing import List, Union

from haystack import Document, component, logging
from haystack.dataclasses import ByteStream

logger = logging.getLogger(__name__)


@component
class ExtensionlessToDocument:
    """
    Lit des fichiers dont le type MIME est inconnu (sans extension)
    et les convertit en Document Haystack comme du texte brut.
    """

    @component.output_types(documents=List[Document])
    def run(self, sources: List[Union[str, Path, ByteStream]]):
        documents = []

        for source in sources:
            try:
                if isinstance(source, ByteStream):
                    raw = source.data
                    meta = source.meta or {}
                else:
                    path = Path(source)
                    raw = path.read_bytes()
                    meta = {"file_path": str(path), "file_name": path.name}

                text = self._decode(raw)

                if text.strip():
                    documents.append(Document(content=text, meta=meta))
                else:
                    logger.warning(
                        "extensionless_converter.empty_content",
                        source=str(source),
                    )

            except Exception as e:
                logger.error(
                    "extensionless_converter.read_error",
                    source=str(source),
                    error=str(e),
                )

        return {"documents": documents}

    @staticmethod
    def _decode(raw: bytes) -> str:
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return raw.decode("latin-1")