import structlog
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from rag_pc4u.core.config import settings
from rag_pc4u.core.document_store import add_List_collection

logger = structlog.get_logger(__name__)

_document_stores: dict[str, QdrantDocumentStore] = {}

def get_document_store(collection_name: str) -> QdrantDocumentStore:
    """Retourne (ou crée) le Document Store Qdrant pour la collection demandée."""
    if collection_name not in _document_stores:
        logger.info(
            "Connexion au QdrantDocumentStore",
            url=settings.qdrant_url,
            collection=collection_name,
        )

        add_List_collection(collection_name)#importrant pour ajouter la collection necessaire

        _document_stores[collection_name] = QdrantDocumentStore(
            url=settings.qdrant_url,
            index=collection_name,
            embedding_dim=settings.embedding_dim,
            use_sparse_embeddings=True,
            recreate_index=False,
        )

    return _document_stores[collection_name]

def make_retriever(collection_name: str):
    """Le retriever sait maintenant dans quelle collection chercher."""
    store = get_document_store(collection_name)
    return QdrantEmbeddingRetriever(document_store=store)