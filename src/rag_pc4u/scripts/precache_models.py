"""Le script se lance UNE SEULE FOIS pour cacher tous les modèles localement."""
import os
from pathlib import Path

# 1. Configuration des chemins et variables d'environnement (À faire en premier)
LOCAL_MODELS_DIR = Path(__file__).parent / "models_cache"
LOCAL_MODELS_DIR.mkdir(exist_ok=True)

os.environ["HF_HOME"]                = str(LOCAL_MODELS_DIR / "hf_cache")
os.environ["FASTEMBED_CACHE_PATH"]   = str(LOCAL_MODELS_DIR / "fastembed_cache")
os.environ["DOCLING_ARTIFACTS_PATH"] = str(LOCAL_MODELS_DIR / "docling_cache")

# 2. Importation des dépendances après configuration de l'environnement
from fastembed import SparseTextEmbedding
from huggingface_hub import snapshot_download
from sentence_transformers import CrossEncoder

print("1/5 — Caching BM25 (FastEmbed)...")
SparseTextEmbedding(model_name="Qdrant/bm25")

print("2/5 — Caching BAAI/bge-m3 tokenizer (Docling HybridChunker)...")
snapshot_download(
    repo_id="BAAI/bge-m3",
    ignore_patterns=["*.safetensors", "*.bin"],
)

print("3/5 — Caching BAAI/bge-reranker-v2-m3 (CrossEncoder)...")
CrossEncoder("BAAI/bge-reranker-v2-m3")

print("4/5 — Caching ds4sd/docling-models (layout PDF, tableaux, figures)...")
# DOCLING_ARTIFACTS_PATH doit être défini AVANT cet import pour que
# docling écrive dans le bon dossier dès l'initialisation.
snapshot_download(repo_id="ds4sd/docling-models")

print("5/5 — Warm-up du pipeline Docling (valide que tout est présent)...")
# Ce warm-up force Docling à initialiser StandardPdfPipeline depuis le cache,
# ce qui confirme que les modèles sont complets avant de passer offline.
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import PdfFormatOption
DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=PdfPipelineOptions(do_ocr=False)
        )
    }
)

print("\n✅ Tous les modèles sont cachés.")
print(f"   HF cache       : {LOCAL_MODELS_DIR / 'hf_cache'}")
print(f"   FastEmbed cache: {LOCAL_MODELS_DIR / 'fastembed_cache'}")
print(f"   Docling cache  : {LOCAL_MODELS_DIR / 'docling_cache'}")
print("\nHF_HUB_OFFLINE=1 peut maintenant être activé.")