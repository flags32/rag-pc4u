import structlog
from haystack.dataclasses.document import Document
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from rag_pc4u.core.config import settings
"""
document_store = QdrantDocumentStore(#fonction qui sert a tester le document store
    ":memory:",#memory signifiant que les données sont stockées en mémoire vive
    recreate_index=True,
    return_embedding=True,
    wait_result_from_api=True,
)
document_store.write_documents(#fonction de test
    [
        Document(content="This is first", embedding=[0.0] * 768),
        Document(content="This is second", embedding=[0.1] * 768),
    ],
)
print(document_store.count_documents())
assert document_store.count_documents() == 2
"""
logger = structlog.get_logger(__name__)

def init_database():
    """Initialise la collection avec le schéma hybride complet (Dense + Sparse)."""
    logger.info("Vérification et création de la collection hybride...")

    # En passant recreate_index=True une première fois quand on lance une démo
    # Qdrant va paramétrer l'espace hybride automatique
    ds = QdrantDocumentStore(
        url=settings.qdrant_url,
        index=settings.collection_name,# probleme à ce niveau car il ne peux pas sortir donc securisé mais je ne sais mais il est possible de faire sortir un client d'une collection car si tout est dans lameme collection pour un seul client cela fait un peu trop pour le model
        embedding_dim=settings.embedding_dim,
        use_sparse_embeddings=True, # Active la configuration du Named Vector "sparse-text" dans Qdrant
        recreate_index=True         # ATTENTION: À passer à False en production pour ne pas vider les données
    )

    logger.info("Collection initialisée avec succès !", total_docs=ds.count_documents())

if __name__ == "__main__":
    init_database()