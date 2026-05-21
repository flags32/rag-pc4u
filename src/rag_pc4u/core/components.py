import structlog
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

from rag_pc4u.core.config import settings

logger = structlog.get_logger(__name__)

# Pattern Singleton propre pour le Document Store
_document_store: QdrantDocumentStore | None = None


def get_document_store() -> QdrantDocumentStore:
    """Initialise de manière unique et sécurisée le Document Store Qdrant."""
    global _document_store
    if _document_store is None:
        logger.info(
            "Connexion au QdrantDocumentStore",
            url=settings.qdrant_url,
            collection=settings.collection_name,
        )
        # Utilisation de l'initialisation recommandée pour Haystack 2.x
        _document_store = QdrantDocumentStore(
            url=settings.qdrant_url,
            index=settings.collection_name,
            embedding_dim=settings.embedding_dim,
            use_sparse_embeddings=True,  # INDISPENSABLE pour le mode Hybride
            recreate_index=False,  # NE PAS recréer à chaque import de composant
        )
    return _document_store



