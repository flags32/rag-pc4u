from typing import List
from pathlib import Path
from haystack import component, Document
from sentence_transformers import CrossEncoder

# Chemin vers le cache local (cohérent avec config.py)
_LOCAL_MODELS_DIR = Path(__file__).parent.parent.parent.parent / "models_cache"
_RERANKER_CACHE = _LOCAL_MODELS_DIR / "hf_cache" / "hub" / "models--BAAI--bge-reranker-v2-m3"


@component
class SimpleBGEReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", top_k: int = 10):
        self.model = CrossEncoder(
            model_name,
            # ✅ Force l'utilisation du cache local uniquement
            local_files_only=True,
        )
        self.top_k = top_k

    @component.output_types(documents=List[Document])
    def run(self, query: str, documents: List[Document]):
        pairs = [(query, doc.content) for doc in documents]
        scores = self.model.predict(pairs)
        scored_docs = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
        return {"documents": [doc for score, doc in scored_docs[:self.top_k]]}