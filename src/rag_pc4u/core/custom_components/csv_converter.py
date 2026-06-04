"""Composant Haystack custom : conversion de fichiers CSV en Documents."""
import csv
from pathlib import Path
from typing import List, Union

import structlog
from haystack import Document, component
from haystack.dataclasses import ByteStream  # <-- 1. Nouvel import indispensable

logger = structlog.get_logger(__name__)


@component
class CSVRowToDocument:
    """
    Convertit un fichier CSV en Documents Haystack — un Document par ligne.
    """

    def __init__(self, encoding: str = "utf-8", delimiter: str = ","):
        self.encoding = encoding
        self.delimiter = delimiter

    @component.output_types(documents=List[Document])
    # 2. Ajout de ByteStream dans la signature
    def run(self, sources: List[Union[str, Path, ByteStream]]) -> dict:
        documents = []

        for source in sources:
            rows_converted = 0

            # 3. Gestion adaptative : si c'est un ByteStream en mémoire ou un fichier physique
            if isinstance(source, ByteStream):
                content_bytes = source.data
                # On récupére le nom depuis les métadonnées du stream s'il y en a
                file_name = source.meta.get("file_name", "stream.csv") if source.meta else "stream.csv"
                file_path = source.meta.get("file_path", file_name) if source.meta else file_name
            else:
                path = Path(source)
                content_bytes = path.read_bytes()
                file_name = path.name
                file_path = str(path)

            try:
                # Fallback latin-1 si utf-8 échoue (fichiers Excel FR fréquents)
                try:
                    text = content_bytes.decode(self.encoding)
                except UnicodeDecodeError:
                    text = content_bytes.decode("latin-1")
                    logger.warning(
                        "csv_converter.encoding_fallback",
                        file=file_name,
                        fallback="latin-1",
                    )

                reader = csv.DictReader(
                    text.splitlines(),
                    delimiter=self.delimiter,
                )

                for i, row in enumerate(reader):
                    # "clé: valeur" séparé par des virgules — lisible par le LLM
                    content = ", ".join(
                        f"{k.strip()}: {v.strip()}"
                        for k, v in row.items()
                        if k and v and v.strip()
                    )
                    if not content:
                        continue  # ligne vide ou que des champs vides

                    documents.append(
                        Document(
                            content=content,
                            meta={
                                "file_path": file_path,
                                "source": file_path,
                                "row_index": i,
                            },
                        )
                    )
                    rows_converted += 1

                logger.info(
                    "csv_converter.converted",
                    file=file_name,
                    rows=rows_converted,
                )

            except Exception as e:
                logger.error(
                    "csv_converter.error",
                    file=file_path,
                    error=str(e),
                )

        return {"documents": documents}