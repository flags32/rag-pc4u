"""Composant Haystack custom : conversion de JSON et XML en texte lisible pour le LLM."""
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Union

import structlog
from haystack import Document, component
from haystack.dataclasses import ByteStream

logger = structlog.get_logger(__name__)


@component
class StructuredDataToDocument:
    """
    Convertit les fichiers JSON et XML en texte plat formaté "Clé: Valeur".
    Supprime la syntaxe informatique inutile ({}, <tags>) pour optimiser les embeddings.
    """

    @component.output_types(documents=List[Document])
    @component.output_types(documents=List[Document])
    def run(self, sources: List[Union[str, Path, ByteStream]]) -> dict:
        documents = []

        for source in sources:
            # 1. Extraction propre de la source et des métadonnées
            if isinstance(source, ByteStream):
                content_bytes = source.data
                file_name = source.meta.get("file_name", "stream.json") if source.meta else "stream.json"
                file_path = source.meta.get("file_path", file_name) if source.meta else file_name
            else:
                path = Path(source)
                content_bytes = path.read_bytes()
                file_name = path.name
                file_path = str(path)

            try:
                text = content_bytes.decode("utf-8", errors="replace")

                # 2. Traitement JSON intelligent (Chunking par objet ou global)
                if file_name.lower().endswith(".json"):
                    try:
                        data = json.loads(text)

                        if isinstance(data, list):
                            # CHUNKING INTELLIGENT : 1 Document indépendant par objet de la liste
                            for i, item in enumerate(data):
                                parsed_item = self._flatten_json(item)
                                if parsed_item.strip():
                                    documents.append(
                                        Document(
                                            content=parsed_item,
                                            meta={
                                                "file_path": file_path,
                                                "source": file_path,
                                                "item_index": i
                                            }
                                        )
                                    )
                        else:
                            # JSON simple (un seul dictionnaire) -> 1 Document global
                            parsed_content = self._flatten_json(data)
                            if parsed_content.strip():
                                documents.append(
                                    Document(
                                        content=parsed_content,
                                        meta={"file_path": file_path, "source": file_path}
                                    )
                                )
                    except json.JSONDecodeError:
                        logger.error("json_converter.parse_error", file=file_path)
                        continue

                # 3. Traitement XML
                elif file_name.lower().endswith(".xml"):
                    try:
                        root = ET.fromstring(text)
                        parsed_content = self._flatten_xml(root)
                        if parsed_content.strip():
                            documents.append(
                                Document(
                                    content=parsed_content,
                                    meta={"file_path": file_path, "source": file_path}
                                )
                            )
                    except ET.ParseError:
                        logger.error("xml_converter.parse_error", file=file_path)
                        continue

            except Exception as e:
                logger.error("structured_converter.error", file=file_path, error=str(e))

        return {"documents": documents}

    def _flatten_json(self, data) -> str:
        """Aplatit un JSON complexe en lignes 'parent > enfant: valeur'"""
        lines = []

        def extract(obj, prefix=""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    extract(v, f"{prefix}{k} > ")
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    extract(v, f"{prefix}[{i}] ")
            else:
                if obj is not None and str(obj).strip():
                    # Nettoie la fin du préfixe pour un affichage propre
                    clean_prefix = prefix[:-3] if prefix.endswith(" > ") else prefix.strip()
                    lines.append(f"{clean_prefix}: {obj}")

        extract(data)
        return "\n".join(lines)

    def _flatten_xml(self, root) -> str:
        """Extrait le texte de toutes les balises XML sous forme 'nom_de_balise: texte'"""
        lines = []
        for elem in root.iter():
            if elem.text and elem.text.strip():
                # Supprime l'éventuel namespace informatique ex: {http://...}Tag
                tag = elem.tag.split('}')[-1]
                lines.append(f"{tag}: {elem.text.strip()}")
        return "\n".join(lines)