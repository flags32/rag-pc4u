# rag_pc4u/core/custom_components/enricher.py
from typing import List
from haystack import component
from haystack.dataclasses import Document


@component
class MetadataEnricher:
    """
    Composant personnalisé Haystack.
    Injecte le client_id dans les métadonnées de chaque document pour garantir le cloisonnement.
    """

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document], client_id: str):
        for doc in documents:
            if doc.meta is None:
                doc.meta = {}
            # On force l'injection du client_id
            doc.meta["client_id"] = client_id

        return {"documents": documents}