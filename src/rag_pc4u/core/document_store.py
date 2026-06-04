"""Utilitaire d'initialisation des collections Qdrant."""
import sys
import structlog
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from rag_pc4u.core.config import settings

logger = structlog.get_logger(__name__)


def init_collection(collection_name: str, recreate: bool = False) -> QdrantDocumentStore:
    """
    Initialise (ou vérifie) une collection Qdrant avec le schéma hybride dense + sparse.

    À appeler une seule fois pour créer la collection.
    En production : recreate=False pour ne jamais vider les données.

    Args:
        collection_name : Nom de la collection Qdrant cible.
        recreate        : True pour repartir de zéro (VIDE TOUTES LES DONNÉES).
    """
    logger.info(
        "Initialisation de la collection",
        collection=collection_name,
        recreate=recreate,
    )
    add_List_collection(collection_name)

    ds = QdrantDocumentStore(
        url=settings.qdrant_url,
        index=collection_name,
        embedding_dim=settings.embedding_dim,
        use_sparse_embeddings=True,
        recreate_index=recreate,
    )
    logger.info(
        "Collection prête",
        collection=collection_name,
        total_docs=ds.count_documents(),
    )
    return ds

def add_List_collection(a):#a represente un nouvelle object de la liste
    if a not in settings.List_collection :
        settings.List_collection.append(a)
    else :
        logger.warning("L'objet est deja dans la liste")

def remove_in_List_collection(a):
    if a in settings.List_collection :
        settings.List_collection.remove(a)
    else :
        logger.warning("L'objet n'est pas dans la liste")


if __name__ == "__main__":
    # Usage : python document_store.py <nom_collection> [--recreate]
    # Exemple : python document_store.py documents_penal --recreate
    target = sys.argv[1] if len(sys.argv) > 1 else settings.default_collection
    should_recreate = "--recreate" in sys.argv
    init_collection(target, recreate=should_recreate)