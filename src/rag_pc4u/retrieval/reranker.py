from typing import List
from haystack import component, Document
from sentence_transformers import CrossEncoder


@component
class SimpleBGEReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", top_k: int = 10):
        self.model = CrossEncoder(model_name)
        self.top_k = top_k

    @component.output_types(documents=List[Document])
    def run(self, query: str, documents: List[Document]):
        # Préparation des paires (query, document)
        pairs = [(query, doc.content) for doc in documents]

        # Calcul des scores par le modèle
        scores = self.model.predict(pairs)

        # Association et tri
        scored_docs = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)

        # Retourne les top_k
        return {"documents": [doc for score, doc in scored_docs[:self.top_k]]}