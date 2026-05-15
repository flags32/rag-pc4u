from haystack.dataclasses.document import Document
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

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
"""ici il n'y a que la version avec la memoire vive """

"""
cependant on peut aussi utiliser cettee version en mémoire persistante :
    from haystack.dataclasses.document import Document
    from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
    from haystack.utils import Secret
    
    document_store = QdrantDocumentStore(
        url="https://XXXXXXXXX.us-east4-0.gcp.cloud.qdrant.io:6333",
        index="your_index_name",
        embedding_dim=1024,  # based on the embedding model
        recreate_index=True,  # enable only to recreate the index and not connect to the existing one
        api_key=Secret.from_token("YOUR_TOKEN"),
    )
    
    document_store.write_documents(
        [
            Document(content="This is first", embedding=[0.0] * 5),
            Document(content="This is second", embedding=[0.1, 0.2, 0.3, 0.4, 0.5]),
        ],
    )
    print(document_store.count_documents())
"""

"""il faut que j'utilise ça QdrantHybridRetriever mais pour ça il me faut un space vector soit un espace vectoriel"""