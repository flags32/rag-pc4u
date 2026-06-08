"""Le script ce lance UNE SEULE FOIS pour cacher tous les modèles localement."""
from pathlib import Path
import os
from fastembed import SparseTextEmbedding
from huggingface_hub import snapshot_download
from sentence_transformers import CrossEncoder

# Même chemin que config.py
LOCAL_MODELS_DIR = Path(__file__).parent / "models_cache"
LOCAL_MODELS_DIR.mkdir(exist_ok=True)
os.environ["HF_HOME"] = str(LOCAL_MODELS_DIR / "hf_cache")
os.environ["FASTEMBED_CACHE_PATH"] = str(LOCAL_MODELS_DIR / "fastembed_cache")

print("1/3 — Caching BM25 (FastEmbed)...")
SparseTextEmbedding(model_name="Qdrant/bm25")

print("2/3 — Caching BAAI/bge-m3 tokenizer (Docling HybridChunker)...")
snapshot_download(
    repo_id="BAAI/bge-m3",
    ignore_patterns=["*.safetensors", "*.bin"],  # tokenizer seul, pas les poids
)

print("3/3 — Caching BAAI/bge-reranker-v2-m3 (CrossEncoder)...")
CrossEncoder("BAAI/bge-reranker-v2-m3")  # télécharge et cache les poids complets

print("Tous les modèles sont cachés. HF_HUB_OFFLINE=1 peut être activé.")