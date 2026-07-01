from typing import List
from pathlib import Path
import torch
from haystack import component, Document
from sentence_transformers import CrossEncoder

# Chemin vers le cache local (cohérent avec config.py)
_LOCAL_MODELS_DIR = Path(__file__).parent.parent.parent.parent / "models_cache"
_RERANKER_CACHE = _LOCAL_MODELS_DIR / "hf_cache" / "hub" / "models--BAAI--bge-reranker-v2-m3"


@component
class SimpleBGEReranker:
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        top_k: int = 10,
        score_threshold: float = 0.2,
    ):
        self.model = CrossEncoder(
            model_name,
            # ✅ Force l'utilisation du cache local uniquement
            local_files_only=True,
            default_activation_function=torch.nn.Sigmoid(),
        )
        self.top_k = top_k
        self.score_threshold = score_threshold

    @component.output_types(documents=List[Document])
    def run(self, query: str, documents: List[Document]):
        pairs = [(query, doc.content) for doc in documents]
        scores = self.model.predict(pairs)
        scored_docs = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)

        # Filtre les documents sous le seuil de confiance et expose le score dans les métadonnées
        filtered = []
        for score, doc in scored_docs:
            if score >= self.score_threshold:
                doc.meta["rerank_score"] = float(score)
                filtered.append(doc)

        # Fallback : retourne au moins le meilleur doc si tout est sous le seuil
        if not filtered:
            best_doc = scored_docs[0][1]
            best_doc.meta["rerank_score"] = float(scored_docs[0][0])
            filtered = [best_doc]

        return {"documents": filtered[:self.top_k]}