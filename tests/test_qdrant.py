import structlog
from haystack.dataclasses import Document
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

# REMPLACE PAR L'IP DE TON CONTENEUR LXC PROXMOX
IP_PROXMOX = "192.168.204.20"

logger = structlog.get_logger(__name__)


def test_connexion_directe():
    print("Connexion à Qdrant sur Proxmox...")

    # On initialise le store en mode sparse uniquement pour bypasser Ollama
    ds = QdrantDocumentStore(
        url=f"http://{IP_PROXMOX}:6333",
        index="collection_test_initial",
        use_sparse_embeddings=True,
        recreate_index=True
    )

    # On crée deux faux documents
    docs = [
        Document(content="Le code secret de la cafétéria PC4U est 1234.", meta={"client_id": "client_demo"}),
        Document(content="Le RAG PC4U est une stack on-premise souveraine.", meta={"client_id": "client_demo"})
    ]

    # On écrit dans ton Proxmox
    ds.write_documents(docs)

    print(f" Succès ! Nombre de documents stockés dans Qdrant : {ds.count_documents()}")


if __name__ == "__main__":
    test_connexion_directe()