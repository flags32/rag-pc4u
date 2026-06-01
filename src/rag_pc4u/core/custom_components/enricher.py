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
            # file_path est injecté explicitement pour garantir son existence dans meta
            # (les convertisseurs Haystack ne le mettent pas tous de façon identique)
            new_meta = {
                **doc.meta,
                "client_id": client_id,
                "file_path": doc.meta.get("file_path") or doc.meta.get("source") or "",
            }
            # On ne passe PAS id=doc.id : l'ID doit être recalculé par Haystack
            # sur le nouveau meta final, sinon l'ID est incohérent avec le contenu stocké
            # ce qui cause des collisions silencieuses dans Qdrant
            enriched.append(
                Document(
                    content=doc.content,
                    meta=new_meta,
                )
            )
        return {"documents": enriched}

    def __repr__(self):
        return "MetadataEnricher()"


