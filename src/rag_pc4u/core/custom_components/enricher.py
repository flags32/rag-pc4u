"""Composant Haystack custom : enrichissement des métadonnées de cloisonnement."""
from typing import List, Optional

from haystack import component, Document


@component
class MetadataEnricher:
    """
    Injecte le client_id dans les métadonnées de chaque document avant indexation.

    Ce composant est indispensable pour le cloisonnement multi-tenant :
    sans lui, les filtres Qdrant par client_id ne trouveront rien.

    Connexion dans le pipeline :
        splitter.documents → enricher.documents
        enricher.documents → dense_embedder.documents
    """

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document], client_id: str) -> dict:
        """
        Ajoute client_id (et éventuellement d'autres champs) dans doc.meta.

        Args:
            documents: Liste de Documents Haystack issus du splitter.
            client_id: Identifiant du client propriétaire des documents.

        Returns:
            dict avec clé 'documents' contenant les documents enrichis.
        """
        enriched = []
        for doc in documents:
            # On ne mutate pas l'objet original
            new_meta = {**doc.meta, "client_id": client_id}
            enriched.append(
                Document(
                    content=doc.content,
                    meta=new_meta,
                    id=doc.id,
                )
            )
        return {"documents": enriched}

    def __repr__(self):
        return f"MetadataEnricher(client_id='{self.client_id}')"
