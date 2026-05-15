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