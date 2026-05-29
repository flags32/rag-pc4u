import structlog
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from rag_pc4u.core.config import settings

logger = structlog.get_logger(__name__)

_document_stores: dict[str, QdrantDocumentStore] = {}

def get_document_store(collection_name: str | None = None) -> QdrantDocumentStore:
    """
    Retourne (ou crée) le Document Store Qdrant pour la collection demandée.

    Si collection_name n'est pas fourni, utilise settings.collection_name
    (comportement mono-tenant ou demo).

    En multi-tenant,il faut passer explicitement le nom de collection dérivé du client_id :
        store = get_document_store(f"documents_{client_id}")
    """
    target_collection = collection_name or settings.collection_name

    if target_collection not in _document_stores:
        logger.info(
            "Connexion au QdrantDocumentStore",
            url=settings.qdrant_url,
            collection=target_collection,
        )
        _document_stores[target_collection] = QdrantDocumentStore(
            url=settings.qdrant_url,
            index=target_collection,
            embedding_dim=settings.embedding_dim,
            use_sparse_embeddings=True,
            recreate_index=False,
        )

    return _document_stores[target_collection]


